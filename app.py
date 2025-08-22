
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Platable Dashboard", layout="wide")

BRAND_PRIMARY = "#E83D57"
CARD_BG = "var(--secondary-background-color)"
TEXT = "var(--text-color)"

st.markdown(f"""
<style>
.kpi {{
  border-radius: 14px;
  padding: 14px;
  background: {CARD_BG};
  border: 1px solid rgba(17,24,39,0.06);
}}
.kpi .label {{ font-size: 0.85rem; opacity: .75; }}
.kpi .value {{ font-size: 1.4rem; font-weight: 700; color: {TEXT}; }}
.section-title {{ font-weight: 700; font-size: 1.05rem; margin: 8px 0 4px; }}
hr.div {{ border: none; border-top: 1px solid rgba(17,24,39,0.06); margin: 8px 0 16px; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#F1F5F9; font-size:.75rem; }}
</style>
""", unsafe_allow_html=True)

# -------------- Helpers --------------
def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.strip().lower())

def parse_hour_from_time(s):
    if pd.isna(s): return np.nan
    s = str(s)
    # Grab the HH:MM prefix safely
    m = re.match(r'^\s*(\d{1,2}):(\d{2})', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    try:
        return pd.to_datetime(s, errors="coerce").hour
    except Exception:
        return np.nan

def normalize_am(v: str) -> str:
    """Map any AM string to one of: vinit, dante, platable."""
    if v is None:
        return "platable"
    s = str(v).strip().lower()
    if "vinit" in s:
        return "vinit"
    if "dante" in s:
        return "dante"
    return "platable"

def build_user_key(row):
    phone = row.get("phone_number")
    if pd.notna(phone) and str(phone).strip() != "":
        return str(phone).strip()
    email = row.get("email")
    if pd.notna(email) and str(email).strip() != "":
        return str(email).strip().lower()
    return None

def bucket_slot(hr):
    if pd.isna(hr): return "Other (22–5)"
    hr = int(hr)
    if 6 <= hr <= 11: return "Morning (6–11)"
    if 12 <= hr <= 16: return "Afternoon (12–16)"
    if 17 <= hr <= 21: return "Evening (17–21)"
    return "Other (22–5)"

def kpi_tile_row(items):
    cols = st.columns(len(items), gap="small")
    for c, (label, value) in zip(cols, items):
        with c:
            st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)

# -------------- Load data --------------
st.sidebar.markdown("### Platable Dashboard")
uploaded = st.sidebar.file_uploader("Upload master order file (CSV or Excel)", type=["csv","xlsx","xls"])

def load_df(file):
    if file is None:
        # As a convenience, allow running without upload by loading a small demo if present
        return None, "Please upload your master file (CSV/XLSX)."
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            import openpyxl  # noqa: F401
            df = pd.read_excel(file)
        return df, f"Uploaded: {file.name}"
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        return None, "Failed to read file."

raw, source_note = load_df(uploaded)

if raw is None:
    st.info("Upload your master CSV/XLSX to get started.")
    st.stop()

df = raw.copy()
df.columns = [norm(c) for c in df.columns]

# Use the exact columns from your file when possible
# Required (from your CSV): date, time_of_order, order_state, order_number, order_value, delivery_fee, total, revenue, payout, brand, store_name, account_manager, phone_number, email, purchase_item_qty
required = ["date","time_of_order","order_state","order_number","order_value","delivery_fee","total","revenue","payout","brand","store_name","account_manager","phone_number","email","purchase_item_qty"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Some expected columns were not found: {missing}. The app will still try to work if alternatives exist.")
# Build created_at with hour
created_at = pd.to_datetime(df.get("date"), errors="coerce")
hours = df.get("time_of_order").apply(parse_hour_from_time) if "time_of_order" in df.columns else np.nan
created_at = pd.to_datetime(created_at.dt.date) + pd.to_timedelta(pd.Series(hours).fillna(0), unit="h")
df["created_at"] = created_at

# State filter (completed only for KPIs)
df["state"] = df.get("order_state", "").astype(str).str.lower()
completed = df[df["state"] == "completed"].copy()

# Inference: pickup if delivery_fee <= 0
completed["delivery_fee_num"] = pd.to_numeric(completed.get("delivery_fee"), errors="coerce").fillna(0.0)
completed["is_pickup"] = completed["delivery_fee_num"] <= 0

# Numerics
for col in ["order_value","delivery_fee","total","revenue","payout","purchase_item_qty"]:
    if col in completed.columns:
        completed[col] = pd.to_numeric(completed[col], errors="coerce").fillna(0.0)
    else:
        completed[col] = 0.0

# Build user key (phone -> email)
completed["user_key"] = completed.apply(build_user_key, axis=1)

# Normalize AM field
if "account_manager" in completed.columns:
    completed["am_norm"] = completed["account_manager"].apply(normalize_am)
else:
    completed["am_norm"] = "platable"

# Dimensions
brands_df  = completed[["brand"]].dropna().drop_duplicates().rename(columns={"brand":"brand_name"})
outlets_df = completed[["store_name","brand"]].dropna().drop_duplicates()
# Vendor column may not exist; detect
vendor_available = "vendor" in completed.columns or "vendor_name" in completed.columns
if vendor_available:
    vcol = "vendor" if "vendor" in completed.columns else "vendor_name"
    vendors_df = completed[[vcol]].dropna().drop_duplicates().rename(columns={vcol:"vendor_name"})
else:
    vendors_df = pd.DataFrame(columns=["vendor_name"])

# ================= Filters =================
# Derive view options: hide Vendor if no vendor column
view_options = ["Company"]
if not vendors_df.empty: view_options.append("Vendor")
view_options.append("Account Manager")
view = st.sidebar.radio("View", options=view_options, horizontal=False)

# Date preset
min_date, max_date = completed["created_at"].min().date(), completed["created_at"].max().date()
preset = st.sidebar.selectbox("Date preset", ["All time","Last 7 days","Last 30 days","Custom"], index=0)
if preset == "All time":
    start_date, end_date = min_date, max_date
elif preset == "Last 7 days":
    end_date = max_date; start_date = max_date - timedelta(days=6)
elif preset == "Last 30 days":
    end_date = max_date; start_date = max_date - timedelta(days=29)
else:
    dr = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(dr, tuple): start_date, end_date = dr
    else: start_date, end_date = min_date, max_date

# Context filters
vendor_sel = None
am_sel = None

if view == "Vendor":
    options = ["All"] + vendors_df["vendor_name"].tolist()
    vendor_label = st.sidebar.selectbox("Vendor", options, index=0)
    if vendor_label != "All":
        vendor_sel = vendor_label

if view == "Account Manager":
    # Only vinit / dante / platable
    am_vals = sorted(completed["am_norm"].dropna().unique().tolist())
    am_options = ["All"] + am_vals
    am_label = st.sidebar.selectbox("Account Manager", am_options, index=0)
    if am_label != "All":
        am_sel = am_label

# ================= Scoped data =================
scope_mask = (completed["created_at"].dt.date >= start_date) & (completed["created_at"].dt.date <= end_date)
if vendor_sel and ("vendor_name" in vendors_df.columns):
    vcol = "vendor" if "vendor" in completed.columns else "vendor_name"
    scope_mask &= (completed[vcol] == vendor_sel)
if am_sel is not None:
    scope_mask &= (completed["am_norm"] == am_sel)

scoped = completed.loc[scope_mask].copy()

# ================= KPIs =================
def fmt_aed(x): 
    try: return f"AED {x:,.2f}"
    except Exception: return f"AED {x}"

total_orders = len(scoped)
gmv = float(scoped["total"].sum())
payout = float(scoped["payout"].sum())
revenue = float(scoped["revenue"].sum())
active_brands = int(scoped["brand"].nunique())
active_outlets = int(scoped["store_name"].nunique())
active_users = int(pd.Series([u for u in scoped["user_key"] if u]).nunique())
repeat_users = int(pd.Series([u for u in scoped["user_key"] if u]).value_counts()[lambda s: s>=2].shape[0]) if active_users else 0
aov = gmv/total_orders if total_orders else 0.0

avg_weight_g, co2_per_kg_food, avg_pickup_km, co2_per_km = 400, 2.5, 6, 0.18
meals_saved = int(scoped["purchase_item_qty"].sum()) if "purchase_item_qty" in scoped.columns else total_orders
food_saved_kg = meals_saved * avg_weight_g / 1000.0
co2_food = food_saved_kg * co2_per_kg_food
pickup_orders = int(scoped["is_pickup"].sum())
co2_last_mile = pickup_orders * avg_pickup_km * co2_per_km
co2_total = co2_food + co2_last_mile

# ================= Header =================
title = view + " View"
st.markdown(f"### {title}")
st.caption(source_note)
if view == "Vendor" and vendor_sel:
    st.markdown(f"<span class='badge'>Vendor: {vendor_sel}</span>", unsafe_allow_html=True)
if view == "Account Manager" and am_sel:
    st.markdown(f"<span class='badge'>Account Manager: {am_sel}</span>", unsafe_allow_html=True)

# KPI tiles
kpi_tile_row([
    ("Total Orders", f"{total_orders:,}"),
    ("GMV", fmt_aed(gmv)),
    ("Payout", fmt_aed(payout)),
    ("Revenue", fmt_aed(revenue)),
    ("Brands Active", f"{active_brands:,}"),
    ("Outlets Active", f"{active_outlets:,}"),
    ("Active Users", f"{active_users:,}"),
])
kpi_tile_row([
    ("Meals Saved", f"{meals_saved:,}"),
    ("Food Saved (kg)", f"{food_saved_kg:.1f}"),
    ("CO₂ Saved (food)", f"{co2_food:.1f} kg"),
    ("CO₂ Saved (last‑mile)", f"{co2_last_mile:.1f} kg"),
    ("Total CO₂ Saved", f"{co2_total:.1f} kg"),
])
kpi_tile_row([
    ("Unique Users", f"{active_users:,}"),
    ("Repeat Users", f"{repeat_users:,}"),
    ("Average Order Value", fmt_aed(aov)),
])

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ================= Charts =================
if not scoped.empty:
    # Orders/GMV over time
    ts = scoped.groupby(scoped["created_at"].dt.date).agg(orders=("order_number","count"), GMV=("total","sum")).reset_index().rename(columns={"created_at":"date","index":"date"})
    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        st.markdown("<div class='section-title'>Orders over time</div>", unsafe_allow_html=True)
        chart = alt.Chart(ts).mark_area(opacity=0.4, line={"color": BRAND_PRIMARY}).encode(x="created_at:O", y="orders:Q")
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>GMV over time</div>", unsafe_allow_html=True)
        chart2 = alt.Chart(ts).mark_area(opacity=0.4, line={"color": BRAND_PRIMARY}).encode(x="created_at:O", y="GMV:Q")
        st.altair_chart(chart2, use_container_width=True)

    with c3:
        st.markdown("<div class='section-title'>Order type split</div>", unsafe_allow_html=True)
        pie = scoped.assign(kind=scoped["is_pickup"].map({True:"Pickup", False:"Delivery"})) \
                    .groupby("kind").size().reset_index(name="count")
        st.altair_chart(alt.Chart(pie).mark_arc().encode(theta="count:Q", color=alt.Color("kind:N", scale=alt.Scale(range=[BRAND_PRIMARY, "#E5E7EB"])), tooltip=["kind","count"]), use_container_width=True)

    # Time-slot buckets (Morning/Afternoon/Evening/Other)
    scoped["hour"] = scoped["created_at"].dt.hour
    slot_df = scoped["hour"].apply(bucket_slot).value_counts().rename_axis("slot").reset_index(name="orders").sort_values("slot")
    st.markdown("<div class='section-title'>Popular time slots</div>", unsafe_allow_html=True)
    st.altair_chart(alt.Chart(slot_df).mark_bar().encode(x=alt.X("slot:N", sort=None, title="Slot"), y="orders:Q", tooltip=["slot","orders"]), use_container_width=True)
else:
    st.info("No data in the selected scope.")

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ================= Tables =================
def filter_table(df: pd.DataFrame, cols: list[str], key: str):
    with st.expander("Filters", expanded=False):
        fcols = st.columns(min(len(cols), 4))
        filters = {}
        for i, col in enumerate(cols):
            if i < len(fcols):
                with fcols[i]:
                    filters[col] = st.text_input(f"{col}", key=f"{key}_{col}")
            else:
                filters[col] = st.text_input(f"{col}", key=f"{key}_{col}")
        mask = pd.Series([True]*len(df))
        for col, q in filters.items():
            if q:
                mask &= df[col].astype(str).str.contains(q, case=False, na=False)
        return df[mask]

if not scoped.empty:
    # Top outlets
    top_outlets = scoped.groupby(["store_name","brand"]).agg(
        Orders=("order_number","count"),
        GMV=("total","sum"),
        AOV=("total","mean"),
        Meals=("purchase_item_qty","sum"),
        PickupPct=("is_pickup","mean"),
        Payout=("payout","sum"),
        Revenue=("revenue","sum"),
    ).reset_index().sort_values("GMV", ascending=False)
    top_outlets["Pickup %"] = (top_outlets["PickupPct"]*100).round(1); top_outlets = top_outlets.drop(columns=["PickupPct"])
    st.markdown("<div class='section-title'>Top outlets by GMV</div>", unsafe_allow_html=True)
    ft = filter_table(top_outlets.rename(columns={"store_name":"Outlet","brand":"Brand"}), ["Outlet","Brand"], key="top_outlets")
    st.dataframe(ft, use_container_width=True, hide_index=True)

    # Top brands
    top_brands = scoped.groupby("brand").agg(
        Orders=("order_number","count"),
        GMV=("total","sum"),
        AOV=("total","mean"),
        PickupPct=("is_pickup","mean"),
        Payout=("payout","sum"),
        Revenue=("revenue","sum"),
    ).reset_index().sort_values("GMV", ascending=False).rename(columns={"brand":"Brand"})
    top_brands["Pickup %"] = (top_brands["PickupPct"]*100).round(1); top_brands = top_brands.drop(columns=["PickupPct"])
    st.markdown("<div class='section-title'>Top brands</div>", unsafe_allow_html=True)
    fb = filter_table(top_brands, ["Brand"], key="top_brands")
    st.dataframe(fb, use_container_width=True, hide_index=True)

    # Top items (if present)
    if "item_name" in scoped.columns:
        top_items = scoped.groupby("item_name").agg(
            Orders=("order_number","count"),
            GMV=("total","sum"),
            AOV=("total","mean"),
        ).reset_index().sort_values("GMV", ascending=False).rename(columns={"item_name":"Item"})
        st.markdown("<div class='section-title'>Top items</div>", unsafe_allow_html=True)
        fi = filter_table(top_items, ["Item"], key="top_items")
        st.dataframe(fi, use_container_width=True, hide_index=True)

# All orders log
log_cols = ["order_number","created_at","state","brand","store_name","purchase_item_qty","order_value","delivery_fee","total","revenue","payout","account_manager"]
avail = [c for c in log_cols if c in scoped.columns]
log = scoped[avail].copy().rename(columns={
    "order_number":"Order #",
    "created_at":"Date/Time",
    "state":"State",
    "brand":"Brand",
    "store_name":"Outlet",
    "purchase_item_qty":"Qty",
    "order_value":"Item Value [AED]",
    "delivery_fee":"Delivery Fee [AED]",
    "total":"Total [AED]",
    "revenue":"Revenue [AED]",
    "payout":"Payout [AED]",
    "account_manager":"Account Manager",
})
if not log.empty:
    log["Date/Time"] = pd.to_datetime(log["Date/Time"]).dt.tz_localize(None)
st.markdown("<div class='section-title'>All orders</div>", unsafe_allow_html=True)
st.dataframe(log, use_container_width=True, hide_index=True)
