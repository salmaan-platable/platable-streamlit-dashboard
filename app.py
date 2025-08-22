import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import io, re, hashlib

st.set_page_config(page_title="Platable Dashboard", layout="wide")

# --------- Theme-aware lightweight CSS (minimal) ----------
BRAND_PRIMARY = "#E83D57"  # Platable primary
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
.kpi .label {{
  font-size: 0.85rem;
  opacity: .75;
}}
.kpi .value {{
  font-size: 1.4rem;
  font-weight: 700;
  color: {TEXT};
}}
.kpi .sub {{
  font-size: 0.75rem;
  opacity: .6;
}}
.section-title {{
  font-weight: 700;
  font-size: 1.05rem;
  margin: 8px 0 4px;
}}
hr.div {{
  border: none; border-top: 1px solid rgba(17,24,39,0.06); margin: 8px 0 16px;
}}
.small {{
  font-size: 0.8rem; opacity: .7;
}}
.badge {{
  display:inline-block; padding:2px 8px; border-radius:999px; background:#F1F5F9; font-size:.75rem;
}}
</style>
""", unsafe_allow_html=True)

# ===================== DATA LOADING =====================
def norm(s: str):
    return re.sub(r'[^a-z0-9]+', '_', s.strip().lower())

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: norm(c) for c in df.columns})
    mapping_candidates = {
        "order_id": ["order_id","order_no","order_number","id"],
        "created_at": ["created_at","datetime","order_date","date_time","date"],
        "time": ["time","order_time"],
        "state": ["state","order_state","status"],
        "is_pickup": ["is_pickup","pickup","type","fulfillment","fulfilment","order_type"],
        "quantity": ["quantity","qty","qnty"],
        "item_value_aed": ["item_value_aed","order_value","amount","value","price"],
        "delivery_fee_aed": ["delivery_fee_aed","delivery_fee","delivery","del_fee"],
        "discount_aed": ["discount_aed","discount"],
        "total_aed": ["total_aed","total","gmv"],
        "platform_revenue_aed": ["platform_revenue_aed","revenue","platform_revenue","commission_amount"],
        "vendor_payout_aed": ["vendor_payout_aed","payout","vendor_payout"],
        "brand_name": ["brand_name","brand"],
        "outlet_name": ["outlet_name","store_name","outlet","restaurant"],
        "vendor_name": ["vendor_name","vendor"],
        "account_manager_name": ["account_manager_name","account_manager","am","manager"],
        "city": ["city","location"],
        "payment_method": ["payment_method","payment","pay_method"],
        "promo_code": ["promo_code","promo","coupon","code"],
        "cancellation_reason": ["cancellation_reason","cancel_reason"],
        "refunded": ["refunded","is_refunded"],
    }
    # Rename to canonical names
    rename_map = {}
    for canon, cands in mapping_candidates.items():
        for c in cands:
            if c in df.columns:
                rename_map[c] = canon
                break
    df = df.rename(columns=rename_map)

    # Build datetime from date + optional time
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"].astype(str), errors="coerce")
    elif "date" in df.columns:
        df["created_at"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    else:
        df["created_at"] = pd.NaT

    if "time" in df.columns and df["time"].notna().any():
        t = pd.to_datetime(df["time"].astype(str), errors="coerce").dt.time
        df["created_at"] = df["created_at"].dt.floor("D")
        df.loc[t.notna(), "created_at"] = df.loc[t.notna(), "created_at"] + pd.to_timedelta(
            pd.to_datetime(df.loc[t.notna(), "time"].astype(str), errors="coerce").dt.hour, unit="h"
        )

    # Normalize state → pending / cancelled / completed (default to completed)
    def norm_state(x):
        s = str(x).strip().lower()
        if "pend" in s: return "pending"
        if "cancel" in s: return "cancelled"
        if s in ["completed","complete","success","done","paid"]: return "completed"
        return "completed"
    df["state"] = df["state"].apply(norm_state) if "state" in df.columns else "completed"

    # is_pickup
    def to_bool_pickup(x):
        s = str(x).strip().lower()
        if s in ["pickup","pick-up","pick up","collection","self_pickup","true","1","yes","y"]:
            return True
        if s in ["delivery","del","false","0","no","n"]:
            return False
        return False
    df["is_pickup"] = df["is_pickup"].apply(to_bool_pickup) if "is_pickup" in df.columns else False

    # Numerics
    for col in ["quantity","item_value_aed","delivery_fee_aed","discount_aed","total_aed","platform_revenue_aed","vendor_payout_aed","commission_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # IDs and names
    if "order_id" not in df.columns:
        df["order_id"] = [f"A{100000+i}" for i in range(len(df))]

    for name_col, id_col in [("brand_name","brand_id"),("outlet_name","outlet_id"),("vendor_name","vendor_id"),("account_manager_name","account_manager_id")]:
        if name_col not in df.columns:
            df[name_col] = ""
        df[id_col] = df[name_col].fillna("").apply(lambda s: (name_col[:2] + hashlib.md5(s.encode()).hexdigest()[:6]) if s else "")

    for col, default in [("city",""),("payment_method",""),("promo_code",""),("cancellation_reason",""),("refunded",False)]:
        if col not in df.columns:
            df[col] = default

    df = df[df["created_at"].notna()].copy()
    return df

def make_mock():
    np.random.seed(3)
    brands = [
        {"brand_id": "b1", "brand_name": "Dunkin'", "vendor_id": "v1"},
        {"brand_id": "b2", "brand_name": "Common Grounds", "vendor_id": "v2"},
        {"brand_id": "b3", "brand_name": "Saya Brasserie", "vendor_id": "v3"},
        {"brand_id": "b4", "brand_name": "MS Cafe - DIFC", "vendor_id": "v4"},
        {"brand_id": "b5", "brand_name": "Krispy Kreme", "vendor_id": "v1"},
    ]
    outlets = [
        {"outlet_id": "o1", "outlet_name": "Dunkin' – Marina", "brand_id": "b1", "city": "Dubai", "account_manager_id": "am1"},
        {"outlet_id": "o2", "outlet_name": "Common Grounds – MOE", "brand_id": "b2", "city": "Dubai", "account_manager_id": "am1"},
        {"outlet_id": "o3", "outlet_name": "Saya – City Walk", "brand_id": "b3", "city": "Dubai", "account_manager_id": "am2"},
        {"outlet_id": "o4", "outlet_name": "MS Cafe – DIFC", "brand_id": "b4", "city": "Dubai", "account_manager_id": "am2"},
        {"outlet_id": "o5", "outlet_name": "Krispy Kreme – Yas Mall", "brand_id": "b5", "city": "Abu Dhabi", "account_manager_id": "am1"},
    ]
    vendors = [
        {"vendor_id": "v1", "vendor_name": "Dunkin' Vendor"},
        {"vendor_id": "v2", "vendor_name": "Common Grounds Vendor"},
        {"vendor_id": "v3", "vendor_name": "Saya Vendor"},
        {"vendor_id": "v4", "vendor_name": "MS Cafe Vendor"},
    ]
    ams = [
        {"account_manager_id": "am1", "account_manager_name": "Vinit Sagar"},
        {"account_manager_id": "am2", "account_manager_name": "Dante Reyes"},
    ]
    # 120 days so the date picker is not limited
    start = (datetime.now().date() - timedelta(days=119))
    dates = pd.date_range(start, periods=120, freq="D")
    rows, oid = [], 1000
    for d in dates:
        for _ in range(np.random.randint(8, 22)):
            brand = np.random.choice(brands)
            outlet = np.random.choice([o for o in outlets if o["brand_id"] == brand["brand_id"]])
            qty = np.random.choice([1,1,1,2,3])
            is_pickup = bool(np.random.choice([True, False], p=[0.65, 0.35]))
            state = np.random.choice(["completed","completed","completed","cancelled","pending"], p=[0.86,0.05,0.02,0.05,0.02])
            item_val = np.random.randint(15, 40)
            deliv = 0 if is_pickup else np.random.choice([5,6,7,8])
            discount = np.random.choice([0,0,0,3,5])
            total = item_val + deliv - discount
            commission_pct = 0.2
            revenue = round(item_val * commission_pct, 2)
            payout = round(total - revenue, 2)
            rows.append({
                "order_id": f"A{oid}", "created_at": d + timedelta(minutes=np.random.randint(9*60, 23*60)),
                "user_id": f"u{np.random.randint(1, 300)}", "brand_id": brand["brand_id"], "outlet_id": outlet["outlet_id"],
                "vendor_id": brand["vendor_id"], "account_manager_id": outlet["account_manager_id"],
                "brand_name": brand["brand_name"], "outlet_name": outlet["outlet_name"],
                "vendor_name": [v for v in vendors if v["vendor_id"]==brand["vendor_id"]][0]["vendor_name"],
                "account_manager_name": [a for a in ams if a["account_manager_id"]==outlet["account_manager_id"]][0]["account_manager_name"],
                "item_name": "Surprise Box", "quantity": qty, "is_pickup": is_pickup, "state": state,
                "item_value_aed": item_val, "delivery_fee_aed": deliv, "discount_aed": discount, "total_aed": total,
                "commission_pct": commission_pct, "platform_revenue_aed": revenue, "vendor_payout_aed": payout,
                "payment_method": np.random.choice(["Card","Apple Pay","Cash"]), "promo_code": np.random.choice(["","NEW15","WELCOME5",""]), "city": outlet["city"],
                "cancellation_reason": "" if state=="completed" else np.random.choice(["user_cancelled","store_cancelled","no_show"]),
                "refunded": False
            })
            oid += 1
    return pd.DataFrame(rows)

# Sidebar: upload
st.sidebar.markdown("### Platable Dashboard")
uploaded = st.sidebar.file_uploader("Upload master order file (CSV or Excel)", type=["csv","xlsx","xls"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        orders = map_columns(raw)
        source_note = f"Uploaded file: {uploaded.name}"
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        orders = make_mock()
        source_note = "Using sample data (failed to parse upload)."
else:
    orders = make_mock()
    source_note = "Using sample data. Upload a file to use your real data."

# Dimensions
brands_df  = orders[["brand_id","brand_name"]].drop_duplicates()
outlets_df = orders[["outlet_id","outlet_name","brand_id","city","account_manager_id"]].drop_duplicates()
vendors_df = orders[["vendor_id","vendor_name"]].drop_duplicates()
ams_df     = orders[["account_manager_id","account_manager_name"]].drop_duplicates()

# ===================== FILTERS =====================
view = st.sidebar.radio("View", options=["Company", "Vendor", "Account Manager"], horizontal=False, key="view_radio")

# Date presets
min_date, max_date = orders["created_at"].min().date(), orders["created_at"].max().date()
preset = st.sidebar.selectbox("Date preset", ["All time","Last 7 days","Last 30 days","Custom"], index=1 if (max_date - min_date).days>30 else 0, key="preset")
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

# Scoped selectors (auto-select first so Vendor/AM actually differ)
vendor_sel = None
am_sel = None
if view == "Vendor":
    vendor_labels = vendors_df["vendor_name"].fillna("").tolist()
    vendor_label = st.sidebar.selectbox("Vendor", ["-- pick vendor --"] + vendor_labels, index=1 if vendor_labels else 0, key="vendor_select")
    if vendor_label != "-- pick vendor --":
        vendor_sel = vendors_df.loc[vendors_df["vendor_name"]==vendor_label, "vendor_id"].values[0]
elif view == "Account Manager":
    am_labels = ams_df["account_manager_name"].fillna("").tolist()
    am_label = st.sidebar.selectbox("Account Manager", ["-- pick manager --"] + am_labels, index=1 if am_labels else 0, key="am_select")
    if am_label != "-- pick manager --":
        am_sel = ams_df.loc[ams_df["account_manager_name"]==am_label, "account_manager_id"].values[0]

# Impact assumptions
st.sidebar.markdown("---")
st.sidebar.markdown("#### Impact settings")
avg_weight_g     = st.sidebar.number_input("Avg weight per meal (g)", value=400, step=25, key="avg_w")
co2_per_kg_food  = st.sidebar.number_input("CO₂ per kg food", value=2.5, step=0.1, key="co2_food")
avg_pickup_km    = st.sidebar.number_input("Avg pickup km saved", value=6.0, step=0.5, key="km_saved")
co2_per_km       = st.sidebar.number_input("CO₂ per km", value=0.18, step=0.01, key="co2_km")

# ===================== CORE UTILS =====================
def scoped_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)]
    if view == "Vendor" and vendor_sel:
        df = df[df["vendor_id"] == vendor_sel]
    if view == "Account Manager" and am_sel:
        am_outlets = outlets_df.loc[outlets_df["account_manager_id"]==am_sel, "outlet_id"]
        df = df[df["outlet_id"].isin(am_outlets)]
    return df

def kpis(df: pd.DataFrame):
    completed = df[df["state"]=="completed"]
    total_orders = len(completed)
    gmv = float(completed["total_aed"].sum())
    payout = float(completed["vendor_payout_aed"].sum())
    revenue = float(completed["platform_revenue_aed"].sum())
    active_brands = completed["brand_id"].nunique()
    active_outlets = completed["outlet_id"].nunique()
    active_users = completed["user_id"].nunique() if "user_id" in completed.columns else 0

    meals_saved = int(completed["quantity"].sum()) if "quantity" in completed.columns else total_orders
    food_saved_kg = meals_saved * avg_weight_g / 1000.0
    co2_food = food_saved_kg * co2_per_kg_food
    pickup_orders = int(completed["is_pickup"].sum())
    co2_last_mile = pickup_orders * avg_pickup_km * co2_per_km
    co2_total = co2_food + co2_last_mile

    users_counts = completed.groupby("user_id").size() if "user_id" in completed.columns else pd.Series([], dtype=int)
    unique_users = int(users_counts.shape[0]) if not users_counts.empty else 0
    repeat_users = int((users_counts >= 2).sum()) if not users_counts.empty else 0
    aov = (gmv / total_orders) if total_orders else 0.0

    return {
        "total_orders": total_orders, "gmv": gmv, "payout": payout, "revenue": revenue,
        "active_brands": active_brands, "active_outlets": active_outlets, "active_users": active_users,
        "meals_saved": meals_saved, "food_saved_kg": food_saved_kg, "co2_food": co2_food,
        "co2_last_mile": co2_last_mile, "co2_total": co2_total,
        "unique_users": unique_users, "repeat_users": repeat_users, "aov": aov
    }

def fmt_aed(x):
    try: return f"AED {x:,.0f}"
    except Exception: return f"AED {x}"

def kpi_grid(row):
    cols = st.columns(len(row), gap="small")
    for c, (label, value, sub) in zip(cols, row):
        with c:
            st.markdown(
                f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>"
                + (f"<div class='sub'>{sub}</div>" if sub else "")
                + "</div>", unsafe_allow_html=True
            )

# ===================== MAIN =====================
scoped = scoped_orders(orders)
K = kpis(scoped)

title = "Company" if view=="Company" else ("Vendor" if view=="Vendor" else "Account Manager")
st.markdown(f"### {title} View")
st.caption(source_note)
if view == "Vendor" and vendor_sel:
    vendor_name = vendors_df.loc[vendors_df["vendor_id"]==vendor_sel, "vendor_name"].values[0]
    st.markdown(f"<span class='badge'>Vendor: {vendor_name}</span>", unsafe_allow_html=True)
if view == "Account Manager" and am_sel:
    am_name = ams_df.loc[ams_df["account_manager_id"]==am_sel, "account_manager_name"].values[0]
    st.markdown(f"<span class='badge'>Account Manager: {am_name}</span>", unsafe_allow_html=True)

# KPI rows
kpi_grid([
    ("Total Orders", f"{K['total_orders']:,}", ""),
    ("GMV", fmt_aed(K["gmv"]), ""),
    ("Payout", fmt_aed(K["payout"]), ""),
    ("Revenue", fmt_aed(K["revenue"]), ""),
    ("Brands Active", f"{K['active_brands']:,}", ""),
    ("Outlets Active", f"{K['active_outlets']:,}", ""),
    ("Active Users", f"{K['active_users']:,}", ""),
])
kpi_grid([
    ("Meals Saved", f"{K['meals_saved']:,}", ""),
    ("Food Saved (kg)", f"{K['food_saved_kg']:.1f}", ""),
    ("CO₂ Saved (food)", f"{K['co2_food']:.1f} kg", ""),
    ("CO₂ Saved (last-mile)", f"{K['co2_last_mile']:.1f} kg", ""),
    ("Total CO₂ Saved", f"{K['co2_total']:.1f} kg", ""),
])
kpi_grid([
    ("Unique Users", f"{K['unique_users']:,}", ""),
    ("Repeat Users", f"{K['repeat_users']:,}", ""),
    ("Average Order Value", fmt_aed(K["aov"]), ""),
])

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# Charts
completed = scoped[scoped["state"]=="completed"].copy()
if not completed.empty:
    completed["date"] = completed["created_at"].dt.date
    series = completed.groupby("date").agg(orders=("order_id","count"), GMV=("total_aed","sum")).reset_index()

    c1, c2, c3 = st.columns([1,1,1], gap="small")

    with c1:
        st.markdown("<div class='section-title'>Orders over time</div>", unsafe_allow_html=True)
        chart = alt.Chart(series).mark_area(opacity=0.4, line={"color": BRAND_PRIMARY}).encode(
            x="date:T", y=alt.Y("orders:Q")
        ).properties(height=230)
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>GMV over time</div>", unsafe_allow_html=True)
        chart2 = alt.Chart(series).mark_area(opacity=0.4, line={"color": BRAND_PRIMARY}).encode(
            x="date:T", y=alt.Y("GMV:Q")
        ).properties(height=230)
        st.altair_chart(chart2, use_container_width=True)

    with c3:
        st.markdown("<div class='section-title'>Order type split</div>", unsafe_allow_html=True)
        pie = completed.assign(kind=completed["is_pickup"].map({True:"Pickup", False:"Delivery"})) \
                       .groupby("kind").size().reset_index(name="count")
        chart3 = alt.Chart(pie).mark_arc().encode(
            theta="count:Q",
            color=alt.Color("kind:N", scale=alt.Scale(range=[BRAND_PRIMARY, "#E5E7EB"])),
            tooltip=["kind","count"]
        ).properties(height=230)
        st.altair_chart(chart3, use_container_width=True)

    st.markdown("<div class='section-title'>Popular time slots (hourly)</div>", unsafe_allow_html=True)
    hours = completed.copy()
    hours["hour"] = hours["created_at"].dt.hour
    hour_df = hours.groupby("hour").size().reset_index(name="orders")
    bar = alt.Chart(hour_df).mark_bar().encode(
        x=alt.X("hour:O", title="Hour of day"),
        y="orders:Q",
        tooltip=["hour","orders"]
    ).properties(height=280)
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("No completed orders in the selected range.")

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# --------- Helper: smart filter UI for tables ----------
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

# Tables (scoped)
if not completed.empty:
    # Top outlets
    top_outlets = completed.groupby(["outlet_id","brand_id"]).agg(
        orders=("order_id","count"),
        GMV=("total_aed","sum"),
        AOV=("total_aed","mean"),
        meals=("quantity","sum"),
        pickup_pct=("is_pickup","mean"),
        payout=("vendor_payout_aed","sum"),
        revenue=("platform_revenue_aed","sum"),
    ).reset_index().sort_values("GMV", ascending=False)

    top_outlets = top_outlets.merge(outlets_df[["outlet_id","outlet_name","city"]], on="outlet_id", how="left")
    top_outlets = top_outlets.merge(brands_df[["brand_id","brand_name"]], on="brand_id", how="left")
    top_outlets["pickup_pct"] = (top_outlets["pickup_pct"]*100).round(1)
    st.markdown("<div class='section-title'>Top outlets by GMV</div>", unsafe_allow_html=True)
    ft = filter_table(top_outlets.rename(columns={
        "outlet_name":"Outlet","brand_name":"Brand","orders":"Orders","GMV":"GMV [AED]",
        "AOV":"AOV [AED]","meals":"Meals","pickup_pct":"Pickup %","payout":"Payout [AED]","revenue":"Revenue [AED]","city":"City"
    })[["Outlet","Brand","City","Orders","GMV [AED]","AOV [AED]","Meals","Pickup %","Payout [AED]","Revenue [AED]"]],
    ["Outlet","Brand","City"], key="top_outlets")
    st.dataframe(ft, use_container_width=True, hide_index=True)

    # Top brands
    top_brands = completed.groupby("brand_id").agg(
        orders=("order_id","count"),
        GMV=("total_aed","sum"),
        AOV=("total_aed","mean"),
        pickup_pct=("is_pickup","mean"),
        payout=("vendor_payout_aed","sum"),
        revenue=("platform_revenue_aed","sum"),
    ).reset_index().sort_values("GMV", ascending=False).merge(brands_df, on="brand_id", how="left")
    st.markdown("<div class='section-title'>Top brands</div>", unsafe_allow_html=True)
    fb = filter_table(top_brands.rename(columns={
        "brand_name":"Brand","orders":"Orders","GMV":"GMV [AED]","AOV":"AOV [AED]","pickup_pct":"Pickup %","payout":"Payout [AED]","revenue":"Revenue [AED]"
    })[["Brand","Orders","GMV [AED]","AOV [AED]","Payout [AED]","Revenue [AED]"]],
    ["Brand"], key="top_brands")
    st.dataframe(fb, use_container_width=True, hide_index=True)

    # Top items
    if "item_name" in completed.columns:
        top_items = completed.groupby("item_name").agg(
            orders=("order_id","count"),
            GMV=("total_aed","sum"),
            AOV=("total_aed","mean"),
        ).reset_index().sort_values("GMV", ascending=False)
        st.markdown("<div class='section-title'>Top items</div>", unsafe_allow_html=True)
        fi = filter_table(top_items.rename(columns={
            "item_name":"Item","orders":"Orders","GMV":"GMV [AED]","AOV":"AOV [AED]"
        })[["Item","Orders","GMV [AED]","AOV [AED]"]],
        ["Item"], key="top_items")
        st.dataframe(fi, use_container_width=True, hide_index=True)

# All orders (log)
log = scoped.merge(outlets_df[["outlet_id","outlet_name","account_manager_id","city"]], on="outlet_id", how="left") \
           .merge(brands_df[["brand_id","brand_name","vendor_id"]], on="brand_id", how="left") \
           .merge(vendors_df, on="vendor_id", how="left") \
           .merge(ams_df, on="account_manager_id", how="left")

log_disp = log.rename(columns={
    "order_id":"Order #", "created_at":"Date/Time", "state":"State", "is_pickup":"Type(Pickup?)",
    "brand_name":"Brand","vendor_name":"Vendor","outlet_name":"Outlet","item_name":"Item","quantity":"Qty",
    "item_value_aed":"Item Value [AED]","delivery_fee_aed":"Delivery Fee [AED]","discount_aed":"Discount [AED]",
    "total_aed":"Total [AED]","commission_pct":"Commission %","platform_revenue_aed":"Revenue [AED]","vendor_payout_aed":"Payout [AED]",
    "payment_method":"Payment", "promo_code":"Promo", "city":"City", "account_manager_name":"Account Manager",
})
if not log_disp.empty:
    log_disp["Date/Time"] = pd.to_datetime(log_disp["Date/Time"]).dt.tz_localize(None)

st.markdown("<div class='section-title'>All orders</div>", unsafe_allow_html=True)
fo = filter_table(log_disp, ["Order #","Brand","Outlet","Vendor","City","State","Promo","Payment"], key="orders")
st.dataframe(fo, use_container_width=True, hide_index=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.caption("Upload your master CSV/XLSX to drive the dashboard • Views are scoped by Vendor/Account Manager • Brand theme via Streamlit config.")
