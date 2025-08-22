
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Platable Dashboard", layout="wide")

# --------- Theme-aware lightweight CSS (minimal) ----------
BRAND_PRIMARY = "#E83D57"  # Platable raspberry (approx. from Play Store)
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
</style>
""", unsafe_allow_html=True)

# --------- Mock data (replace with DB/CSV) ----------
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

account_managers = [
    {"account_manager_id": "am1", "name": "Vinit Sagar"},
    {"account_manager_id": "am2", "name": "Dante Reyes"},
]

# Create a synthetic orders table over 30 days
start = datetime.now().date() - timedelta(days=29)
dates = pd.date_range(start, periods=30, freq="D")

rows = []
order_id = 1000
for d in dates:
    for _ in range(np.random.randint(10, 25)):  # orders per day
        brand = np.random.choice(brands)
        outlet = np.random.choice([o for o in outlets if o["brand_id"] == brand["brand_id"]])
        qty = np.random.choice([1,1,1,2,3])
        is_pickup = bool(np.random.choice([True, False], p=[0.65, 0.35]))
        state = np.random.choice(["completed","completed","completed","cancelled","pending"], p=[0.8,0.05,0.05,0.05,0.05])
        item_val = np.random.randint(15, 40)
        deliv = 0 if is_pickup else np.random.choice([5,6,7,8])
        discount = np.random.choice([0,0,0,3,5])
        total = item_val + deliv - discount
        commission_pct = 0.2
        revenue = round(item_val * commission_pct, 2)
        payout = round(total - revenue, 2)
        rows.append({
            "order_id": f"A{order_id}", "created_at": d + timedelta(minutes=np.random.randint(9*60, 23*60)),
            "user_id": f"u{np.random.randint(1, 300)}", "brand_id": brand["brand_id"], "outlet_id": outlet["outlet_id"],
            "vendor_id": brand["vendor_id"], "account_manager_id": outlet["account_manager_id"],
            "item_name": "Surprise Box", "quantity": qty, "is_pickup": is_pickup, "state": state,
            "item_value_aed": item_val, "delivery_fee_aed": deliv, "discount_aed": discount, "total_aed": total,
            "commission_pct": commission_pct, "platform_revenue_aed": revenue, "vendor_payout_aed": payout,
            "payment_method": np.random.choice(["Card","Apple Pay","Cash"]), "promo_code": np.random.choice(["","NEW15","WELCOME5",""]), "city": outlet["city"],
            "cancellation_reason": "" if state=="completed" else np.random.choice(["user_cancelled","store_cancelled","no_show"]),
            "refunded": False
        })
        order_id += 1

orders = pd.DataFrame(rows)
brands_df = pd.DataFrame(brands)
outlets_df = pd.DataFrame(outlets)
vendors_df = pd.DataFrame(vendors)
ams_df = pd.DataFrame(account_managers)

# --------- Sidebar filters ----------
st.sidebar.markdown("### Platable Dashboard")
view = st.sidebar.radio("View", options=["Company", "Vendor", "Account Manager"], horizontal=False)

# Date range
min_date, max_date = orders["created_at"].min().date(), orders["created_at"].max().date()
dr = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(dr, tuple):
    start_date, end_date = dr
else:
    start_date, end_date = min_date, max_date

# Impact assumptions
st.sidebar.markdown("---")
st.sidebar.markdown("#### Impact settings")
avg_weight_g = st.sidebar.number_input("Avg weight per meal (g)", value=400, step=25)
co2_per_kg_food = st.sidebar.number_input("CO₂ per kg food", value=2.5, step=0.1)
avg_pickup_km = st.sidebar.number_input("Avg pickup km saved", value=6.0, step=0.5)
co2_per_km = st.sidebar.number_input("CO₂ per km", value=0.18, step=0.01)

# Scoped filters
vendor_sel = None
am_sel = None
if view == "Vendor":
    vendor_options = ["All"] + vendors_df["vendor_name"].tolist()
    vendor_label = st.sidebar.selectbox("Vendor", vendor_options, index=0)
    vendor_sel = None if vendor_label=="All" else vendors_df.set_index("vendor_name").loc[vendor_label, "vendor_id"]
elif view == "Account Manager":
    am_options = ["All"] + ams_df["name"].tolist()
    am_label = st.sidebar.selectbox("Account Manager", am_options, index=0)
    am_sel = None if am_label=="All" else ams_df.set_index("name").loc[am_label, "account_manager_id"]

# --------- Helper functions ----------
def scoped_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)]
    if view == "Vendor" and vendor_sel:
        df = df[df["vendor_id"] == vendor_sel]
    if view == "Account Manager" and am_sel:
        # scope by AM's outlets
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
    active_users = completed["user_id"].nunique()

    meals_saved = int(completed["quantity"].sum())
    food_saved_kg = meals_saved * avg_weight_g / 1000.0
    co2_food = food_saved_kg * co2_per_kg_food
    pickup_orders = int(completed["is_pickup"].sum())
    co2_last_mile = pickup_orders * avg_pickup_km * co2_per_km
    co2_total = co2_food + co2_last_mile

    users_counts = completed.groupby("user_id").size()
    unique_users = int(users_counts.shape[0])
    repeat_users = int((users_counts >= 2).sum())
    aov = (gmv / total_orders) if total_orders else 0.0

    return {
        "total_orders": total_orders, "gmv": gmv, "payout": payout, "revenue": revenue,
        "active_brands": active_brands, "active_outlets": active_outlets, "active_users": active_users,
        "meals_saved": meals_saved, "food_saved_kg": food_saved_kg, "co2_food": co2_food,
        "co2_last_mile": co2_last_mile, "co2_total": co2_total,
        "unique_users": unique_users, "repeat_users": repeat_users, "aov": aov
    }

def fmt_aed(x): 
    try:
        return f"AED {x:,.0f}"
    except Exception:
        return f"AED {x}"

# --------- Render KPI tiles ----------
def kpi_grid(row: list[tuple[str, str, str]]):
    cols = st.columns(len(row), gap="small")
    for c, (label, value, sub) in zip(cols, row):
        with c:
            st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>" + (f"<div class='sub'>{sub}</div>" if sub else "") + "</div>", unsafe_allow_html=True)

# --------- Data scope ----------
scoped = scoped_orders(orders)
K = kpis(scoped)

# --------- Head ---------
title = "Company" if view=="Company" else ("Vendor" if view=="Vendor" else "Account Manager")
st.markdown(f"### {title} View")

# --------- KPI rows ----------
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
    ("CO₂ Saved (last‑mile)", f"{K['co2_last_mile']:.1f} kg", ""),
    ("Total CO₂ Saved", f"{K['co2_total']:.1f} kg", ""),
])

kpi_grid([
    ("Unique Users", f"{K['unique_users']:,}", ""),
    ("Repeat Users", f"{K['repeat_users']:,}", ""),
    ("Average Order Value", fmt_aed(K["aov"]), ""),
])

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# --------- Charts ----------
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
        chart3 = alt.Chart(pie).mark_arc().encode(theta="count:Q", color=alt.Color("kind:N", scale=alt.Scale(range=[BRAND_PRIMARY, "#E5E7EB"])),
                                                  tooltip=["kind","count"])\
                                .properties(height=230)
        st.altair_chart(chart3, use_container_width=True)

    st.markdown("<div class='section-title'>Popular time slots (hourly)</div>", unsafe_allow_html=True)
    hours = completed.copy()
    hours["hour"] = hours["created_at"].dt.hour
    hour_df = hours.groupby("hour").size().reset_index(name="orders")
    bar = alt.Chart(hour_df).mark_bar().encode(x=alt.X("hour:O", title="Hour of day"),
                                               y="orders:Q",
                                               tooltip=["hour","orders"]).properties(height=280)
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("No completed orders in the selected range.")

st.markdown("<hr class='div'/>", unsafe_allow_html=True)

# --------- Helper: smart filter UI for any table ----------
def filter_table(df: pd.DataFrame, cols: list[str], key: str):
    with st.expander("Filters", expanded=False):
        fcols = st.columns(len(cols))
        filters = {}
        for c, col in zip(fcols, cols):
            with c:
                filters[col] = st.text_input(f"{col}", key=f"{key}_{col}")
        mask = pd.Series([True]*len(df))
        for col, q in filters.items():
            if q:
                mask &= df[col].astype(str).str.contains(q, case=False, na=False)
        return df[mask]

# --------- Tables ----------
# Top outlets
top_outlets = completed.groupby("outlet_id").agg(
    orders=("order_id","count"),
    GMV=("total_aed","sum"),
    AOV=("total_aed","mean"),
    meals=("quantity","sum"),
    pickup_pct=("is_pickup","mean"),
    payout=("vendor_payout_aed","sum"),
    revenue=("platform_revenue_aed","sum"),
).reset_index().sort_values("GMV", ascending=False)

if not top_outlets.empty:
    top_outlets = top_outlets.merge(outlets_df[["outlet_id","outlet_name","brand_id","city"]], on="outlet_id", how="left")
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
    "payment_method":"Payment", "promo_code":"Promo", "city":"City", "name":"Account Manager",
})
log_disp["Date/Time"] = pd.to_datetime(log_disp["Date/Time"]).dt.tz_localize(None)

st.markdown("<div class='section-title'>All orders</div>", unsafe_allow_html=True)
fo = filter_table(log_disp, ["Order #","Brand","Outlet","Vendor","City","State","Promo","Payment"], key="orders")
st.dataframe(fo, use_container_width=True, hide_index=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.caption("Brand colors applied via Streamlit theme and minimal CSS. Replace mock data with your warehouse/DB.")

