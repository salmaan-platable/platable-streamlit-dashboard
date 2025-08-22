import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Platable Dashboard", layout="wide")

# -------- Brand styling & sidebar header --------
BRAND_PRIMARY = "#E83D57"
CARD_BG = "var(--secondary-background-color)"
TEXT = "var(--text-color)"

st.markdown(f"""
<style>
/* Cards */
.kpi {{ border-radius: 14px; padding: 14px; background: {CARD_BG}; border: 1px solid rgba(17,24,39,0.06); }}
.kpi .label {{ font-size: 0.85rem; opacity: .75; }}
.kpi .value {{ font-size: 1.4rem; font-weight: 700; color: {TEXT}; }}

.section-title {{ font-weight: 700; font-size: 1.05rem; margin: 8px 0 4px; }}
hr.div {{ border: none; border-top: 1px solid rgba(17,24,39,0.06); margin: 8px 0 16px; }}

/* Sidebar header */
.sidebar-header {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
.logo-circle {{ width:28px; height:28px; border-radius:50%; background:{BRAND_PRIMARY}; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; }}
.brand-name {{ font-weight:800; letter-spacing:.3px; }}

/* Pill radio buttons */
div[role="radiogroup"] label {{ 
  border:1px solid rgba(17,24,39,0.15); 
  padding:6px 12px; 
  border-radius:999px; 
  margin-right:6px; 
  background:white; 
}}
div[role="radiogroup"] label:hover {{ border-color:{BRAND_PRIMARY}; }}
div[role="radiogroup"] label[data-checked="true"] {{
  background:{BRAND_PRIMARY}; color:white; border-color:{BRAND_PRIMARY};
}}
/* reduce default sidebar padding a bit */
section[data-testid="stSidebar"] > div {{ padding-top: 10px; }}
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.strip().lower())

def parse_hour_from_time(s):
    if pd.isna(s): return np.nan
    s = str(s)
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
    if v is None: return "platable"
    s = str(v).strip().lower()
    if "vinit" in s: return "vinit"
    if "dante" in s: return "dante"
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
            st.markdown(
                f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
                unsafe_allow_html=True
            )

# ---------- Sidebar ----------
with st.sidebar:
    # Logo/title
    st.markdown(
        '<div class="sidebar-header"><div class="logo-circle">P</div><div class="brand-name">Platable Dashboard</div></div>',
        unsafe_allow_html=True
    )

# Settings expander: upload + impact parameters
with st.sidebar.expander("Settings", expanded=True):
    uploaded = st.file_uploader("Upload master order file (CSV or Excel)", type=["csv","xlsx","xls"])
    st.markdown("**Impact parameters**")
    avg_weight_g = st.number_input("Avg weight per meal (g)", value=400, step=25, key="avg_w")
    co2_per_kg_food = st.number_input("CO₂ per kg food", value=2.5, step=0.1, key="co2_food")
    avg_pickup_km = st.number_input("Avg pickup km saved", value=6.0, step=0.5, key="km_saved")
    co2_per_km = st.number_input("CO₂ per km", value=0.18, step=0.01, key="co2_km")

# View pills & preset placeholders
view = st.sidebar.radio("View", options=["Company", "Brand", "Account Manager"], horizontal=True)
preset_placeholder = st.sidebar.empty()

# ---------- data load ----------
def load_df(file):
    if file is None:
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
    st.info("Upload your master CSV/XLSX in **Settings** to get started.")
    st.stop()

df = raw.copy()
df.columns = [norm(c) for c in df.columns]

required = ["date","time_of_order","order_state","order_number","order_value","delivery_fee","total","revenue","payout","brand","store_name","account_manager","phone_number","email","purchase_item_qty"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Some expected columns were not found: {missing}. The app will still try to run.")

# timestamps
created_at = pd.to_datetime(df.get("date"), errors="coerce")
hours = df.get("time_of_order").apply(parse_hour_from_time) if "time_of_order" in df.columns else np.nan
created_at = pd.to_datetime(created_at.dt.date) + pd.to_timedelta(pd.Series(hours).fillna(0), unit="h")
df["created_at"] = created_at

# states / pickup
df["state"] = df.get("order_state", "").astype(str).str.lower()
df["delivery_fee_num"] = pd.to_numeric(df.get("delivery_fee", 0), errors="coerce").fillna(0.0)
df["is_pickup"] = df["delivery_fee_num"] <= 0

# user + account manager (on full df)
df["user_key"] = df.apply(build_user_key, axis=1)
df["am_norm"] = df.get("account_manager", "").apply(normalize_am)

# completed scope for ops/finance
completed = df[df["state"] == "completed"].copy()

# date presets
min_date, max_date = pd.to_datetime(df["created_at"]).min().date(), pd.to_datetime(df["created_at"]).max().date()
with preset_placeholder:
    preset = st.selectbox("Date preset", ["All time","Last 7 days","Last 30 days","Custom"], index=0, key="preset")

if preset == "All time":
    start_date, end_date = min_date, max_date
elif preset == "Last 7 days":
    end_date = max_date; start_date = max_date - timedelta(days=6)
elif preset == "Last 30 days":
    end_date = max_date; start_date = max_date - timedelta(days=29)
else:
    dr = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="daterange")
    if isinstance(dr, tuple): start_date, end_date = dr
    else: start_date, end_date = min_date, max_date

# context filters
brand_sel = None
am_sel = None
if view == "Brand":
    brands_df = completed[["brand"]].dropna().drop_duplicates().sort_values("brand")
    bopts = ["All"] + brands_df["brand"].tolist()
    blabel = st.sidebar.selectbox("Brand", bopts, index=0)
    if blabel != "All": brand_sel = blabel

if view == "Account Manager":
    am_opts = ["All"] + sorted(df["am_norm"].dropna().unique().tolist())
    alabel = st.sidebar.selectbox("Account Manager", am_opts, index=0)
    if alabel != "All": am_sel = alabel

# scopes
mask_completed = (completed["created_at"].dt.date >= start_date) & (completed["created_at"].dt.date <= end_date)
if brand_sel: mask_completed &= (completed["brand"] == brand_sel)
if am_sel is not None: mask_completed &= (completed["am_norm"] == am_sel)
scoped = completed.loc[mask_completed].copy()

mask_full = (df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)
if brand_sel: mask_full &= (df["brand"] == brand_sel)
if am_sel is not None: mask_full &= (df["am_norm"] == am_sel)

# ================= KPIs (your definitions) =================
def fmt_aed(x):
    try: return f"AED {x:,.2f}"
    except Exception: return f"AED {x}"

# Total Users (all-time, completed+pending) — ignores current filters
users_alltime_df = df[df["state"].isin(["completed","pending"])]
total_users_alltime = int(pd.Series([u for u in users_alltime_df["user_key"] if u]).nunique()) if not users_alltime_df.empty else 0

# Users Ordered + Repeat Users — current filters, states in {completed, cancelled}
scoped_users_df = df[mask_full & df["state"].isin(["completed","cancelled"])]
scoped_users_series = pd.Series([u for u in scoped_users_df["user_key"] if u])
users_ordered_scope = int(scoped_users_series.nunique()) if not scoped_users_series.empty else 0
repeat_users_scope = int(scoped_users_series.value_counts()[lambda s: s>=2].shape[0]) if not scoped_users_series.empty else 0

# Ops/finance from completed-only scope
total_orders = len(scoped)
gmv = float(pd.to_numeric(scoped["total"], errors="coerce").sum())
payout = float(pd.to_numeric(scoped["payout"], errors="coerce").sum())
revenue = float(pd.to_numeric(scoped["revenue"], errors="coerce").sum())
active_brands = int(scoped["brand"].nunique())
active_outlets = int(scoped["store_name"].nunique())
aov = gmv/total_orders if total_orders else 0.0

# Impact
meals_saved = int(pd.to_numeric(scoped.get("purchase_item_qty", 0), errors="coerce").fillna(0).sum()) if not scoped.empty else 0
food_saved_kg = meals_saved * avg_weight_g / 1000.0
co2_food = food_saved_kg * co2_per_kg_food
pickup_orders = int(scoped["is_pickup"].sum()) if "is_pickup" in scoped.columns else 0
co2_last_mile = pickup_orders * avg_pickup_km * co2_per_km
co2_total = co2_food + co2_last_mile

# ================= header =================
st.markdown(f"### {view} View")
st.caption(source_note)
if brand_sel: st.markdown(f"<span class='badge'>Brand: {brand_sel}</span>", unsafe_allow_html=True)
if am_sel: st.markdown(f"<span class='badge'>Account Manager: {am_sel}</span>", unsafe_allow_html=True)

# KPIs
kpi_tile_row([
    ("Total Orders", f"{total_orders:,}"),
    ("GMV", fmt_aed(gmv)),
    ("Payout", fmt_aed(payout)),
    ("Revenue", fmt_aed(revenue)),
    ("Brands Active", f"{active_brands:,}"),
    ("Outlets Active", f"{active_outlets:,}"),
    ("Users Ordered", f"{users_ordered_scope:,}"),
])
kpi_tile_row([
    ("Meals Saved", f"{meals_saved:,}"),
    ("Food Saved (kg)", f"{food_saved_kg:.1f}"),
    ("CO₂ Saved (food)", f"{co2_food:.1f} kg"),
    ("CO₂ Saved (last-mile)", f"{co2_last_mile:.1f} kg"),
    ("Total CO₂ Saved", f"{co2_total:.1f} kg"),
])
kpi_tile_row([
    ("Total Users (all-time, completed+pending)", f"{total_users_alltime:,}"),
    ("Users Ordered (completed+cancelled)", f"{users_ordered_scope:,}"),
    ("Repeat Users (current filters)", f"{repeat_users_scope:,}"),
])

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ================= growth chart: cumulative monthly indexed line =================
if not scoped.empty:
    ts = scoped.copy()
    ts["month"] = pd.to_datetime(ts["created_at"]).dt.to_period("M").dt.to_timestamp()

    monthly = ts.groupby("month").agg(
        orders=("order_number","count"),
        gmv=("total","sum")
    ).reset_index().sort_values("month")

    monthly["cum_orders"] = monthly["orders"].cumsum()
    monthly["cum_gmv"] = monthly["gmv"].cumsum()

    base_orders = monthly.loc[monthly["cum_orders"]>0, "cum_orders"].iloc[0] if (monthly["cum_orders"]>0).any() else 1.0
    base_gmv = monthly.loc[monthly["cum_gmv"]>0, "cum_gmv"].iloc[0] if (monthly["cum_gmv"]>0).any() else 1.0
    monthly["orders_index"] = 100 * monthly["cum_orders"] / base_orders
    monthly["gmv_index"]    = 100 * monthly["cum_gmv"] / base_gmv

    tidy = pd.melt(
        monthly,
        id_vars=["month","cum_orders","cum_gmv"],
        value_vars=["orders_index","gmv_index"],
        var_name="metric",
        value_name="index_value"
    )
    tidy["metric"] = tidy["metric"].map({"orders_index":"Orders (indexed)","gmv_index":"GMV (indexed)"})
    chart = alt.Chart(tidy).mark_line().encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("index_value:Q", title="Cumulative growth (index = 100 at start)"),
        color=alt.Color("metric:N", scale=alt.Scale(range=[BRAND_PRIMARY, "#111827"])),
        tooltip=[
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("index_value:Q", title="Index", format=".1f"),
            alt.Tooltip("cum_orders:Q", title="Cum Orders", format=",.0f"),
            alt.Tooltip("cum_gmv:Q", title="Cum GMV (AED)", format=",.0f"),
        ]
    ).properties(height=320)
    st.markdown("<div class='section-title'>Cumulative growth (Orders & GMV, monthly indexed)</div>", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)

    # Popular time slots (keep your preferred buckets)
    scoped["hour"] = pd.to_datetime(scoped["created_at"]).dt.hour
    slot_df = scoped["hour"].apply(bucket_slot)\
        .value_counts()\
        .reindex(["Morning (6–11)","Afternoon (12–16)","Evening (17–21)","Other (22–5)"], fill_value=0)\
        .rename_axis("slot").reset_index(name="orders")
    st.markdown("<div class='section-title'>Popular time slots</div>", unsafe_allow_html=True)
    st.altair_chart(
        alt.Chart(slot_df).mark_bar().encode(
            x=alt.X("slot:N", sort=None, title="Slot"),
            y="orders:Q",
            tooltip=["slot","orders"]
        ),
        use_container_width=True
    )
else:
    st.info("No data in the selected scope.")

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# ================= tables =================
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

    if "item_name" in scoped.columns:
        top_items = scoped.groupby("item_name").agg(
            Orders=("order_number","count"),
            GMV=("total","sum"),
            AOV=("total","mean"),
        ).reset_index().sort_values("GMV", ascending=False).rename(columns={"item_name":"Item"})
        st.markdown("<div class='section-title'>Top items</div>", unsafe_allow_html=True)
        fi = filter_table(top_items, ["Item"], key="top_items")
        st.dataframe(fi, use_container_width=True, hide_index=True)

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
