import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent


# Page setup
st.set_page_config(
    page_title="🚖 Uber NCR Ride Insights Dashboard (2024)",
    layout="wide"
)

# Global look & feel
sns.set_theme(style="whitegrid")
ACCENT = "#5B8FF9"
PALETTE = ["#5B8FF9", "#5AD8A6", "#5D7092", "#F6BD16", "#E8684A", "#6DC8EC", "#9270CA"]
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


# Helpers
def pct(x):
    if pd.isna(x) or x==0: 
        return "0%"
    return f"{x*100:.1f}%"

def safe_mean(series):
    if series.dropna().empty:
        return np.nan
    return series.dropna().mean()

def strong(s):  # for bold inline markdown
    return f"**{s}**"

def trend_next_value(y):
    """
    Very lightweight 'forecast':
    - If >= 7 points: last 7-day mean as next value
    - Else if >= 3 points: simple linear fit extrapolation
    - Else: last value as naive forecast
    """
    y = pd.Series(y).dropna()
    if len(y) >= 7:
        return y.tail(7).mean()
    elif len(y) >= 3:
        x = np.arange(len(y))
        a, b = np.polyfit(x, y.values, 1)  # y = a*x + b
        return a*(len(y)) + b
    elif len(y) >= 1:
        return y.iloc[-1]
    return np.nan

def top_k(series, k=3):
    vc = series.value_counts(dropna=True)
    return vc.head(k)


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("ncr_ride_bookings.csv")

    # Dates & time
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["date_only"] = df["Date"].dt.date
    # Parse hour from "Time" like "HH:MM:SS"
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour

    # Week & weekday
    df["Week"] = df["Date"].dt.to_period("W").astype(str)
    df["day_of_week"] = df["Date"].dt.day_name()

    # Outcomes
    df["is_completed"] = (df["Booking Status"].eq("Completed")).astype(int)
    df["is_cancel_driver"] = (df["Cancelled Rides by Driver"] > 0).astype(int)
    df["is_cancel_customer"] = (df["Cancelled Rides by Customer"] > 0).astype(int)
    df["is_incomplete"] = (df["Incomplete Rides"] > 0).astype(int)

    # Short aliases used in filters & visuals
    df["distance"] = pd.to_numeric(df["Ride Distance"], errors="coerce")
    df["revenue"]  = pd.to_numeric(df["Booking Value"], errors="coerce")
    df["driver_rating"] = pd.to_numeric(df["Driver Ratings"], errors="coerce")
    df["cust_rating"]   = pd.to_numeric(df["Customer Rating"], errors="coerce")

    return df

df = load_data()


# Title + Overview
st.title("🚖 Uber NCR Ride Insights Dashboard (2024)")


with st.container():
    st.markdown(dedent(f"""
    #### 📌 What this dashboard shows (using Uber NCR 2024 data)
    - **Demand patterns** by hour and weekday ⏰  
    - **Ride outcomes**: completed vs cancelled (driver / customer) ✅❌  
    - **Distance & completion** relationship 📏  
    - **Cancellation drivers** and **experience (ratings)** ⭐  
    - **Vehicle & payment preferences** 🚗💳  
    - **Revenue trend** 💰  
    """))


# Sidebar Filters
st.sidebar.header("🔍 Filters")

date_min, date_max = df["date_only"].min(), df["date_only"].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=[date_min, date_max],
    min_value=date_min, max_value=date_max
)

if len(date_range) == 2:
    df = df[(df["date_only"] >= date_range[0]) & (df["date_only"] <= date_range[1])]

# Day filter
days_selected = st.sidebar.multiselect(
    "Day of Week",
    options=list(df["day_of_week"].dropna().unique()),
    default=list(df["day_of_week"].dropna().unique())
)
if len(days_selected) > 0:
    df = df[df["day_of_week"].isin(days_selected)]

# Distance filter
dist_min, dist_max = float(df["distance"].min()), float(df["distance"].max())
dist_sel = st.sidebar.slider(
    "Distance Range (km)",
    min_value=float(round(dist_min, 2)),
    max_value=float(round(dist_max, 2)),
    value=(float(round(dist_min, 2)), float(round(dist_max, 2))),
)
df = df[(df["distance"] >= dist_sel[0]) & (df["distance"] <= dist_sel[1])]

# Vehicle filter
veh_all = list(df["Vehicle Type"].dropna().unique())
veh_sel = st.sidebar.multiselect("Vehicle Type", options=veh_all, default=veh_all)
if veh_sel:
    df = df[df["Vehicle Type"].isin(veh_sel)]

# Payment filter
pay_all = list(df["Payment Method"].dropna().unique())
pay_sel = st.sidebar.multiselect("Payment Method", options=pay_all, default=pay_all)
if pay_sel:
    df = df[df["Payment Method"].isin(pay_sel)]

# Quick guard
if df.empty:
    st.warning("No data matches your filters. Please broaden your selections.")
    st.stop()


# KPIs
c1, c2, c3, c4 = st.columns(4)
total_rides = len(df)
comp_rate = safe_mean(df["is_completed"])
cancel_driver = safe_mean(df["is_cancel_driver"])
cancel_customer = safe_mean(df["is_cancel_customer"])
avg_driver_rating = safe_mean(df["driver_rating"])
avg_cust_rating = safe_mean(df["cust_rating"])
total_revenue = df["revenue"].sum(skipna=True)

with c1:
    st.metric("🧾 Total Rides", f"{total_rides:,}")
with c2:
    st.metric("✅ Completion Rate", pct(comp_rate))
with c3:
    st.metric("🚫 Driver Cancels", pct(cancel_driver))
with c4:
    st.metric("🙋 Customer Cancels", pct(cancel_customer))

st.markdown(
    f"> ⭐ Experience snapshot — {strong('Driver Rating:')} "
    f"{avg_driver_rating:.2f} | {strong('Customer Rating:')} {avg_cust_rating:.2f} | "
    f"{strong('Revenue:')} ₹{total_revenue:,.0f}"
)

st.markdown("---")


# 1) Ride Outcomes
st.subheader("1️⃣ Overall Ride Outcomes")
outcomes = df[["is_completed","is_cancel_driver","is_cancel_customer","is_incomplete"]].mean().mul(100).rename({
    "is_completed": "Completed",
    "is_cancel_driver": "Cancel_by Driver",
    "is_cancel_customer": "Cancel_by Customer",
    "is_incomplete": "Incomplete"
})

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=outcomes.index, y=outcomes.values, palette=PALETTE[:4], ax=ax)
ax.set_title("Ride Outcomes (%)")
ax.set_ylabel("Percent")
for i,v in enumerate(outcomes.values):
    ax.text(i, v+0.8, f"{v:.1f}%", ha="center", va="bottom")
st.pyplot(fig)

# Dynamic narrative + prediction
st.markdown(dedent(f"""
- ✅ {strong('Completed')} rides: **{outcomes['Completed']:.1f}%** of filtered trips.  
- ❌ {strong('Driver cancellations')}: **{outcomes['Cancel_by Driver']:.1f}%** — could indicate mismatch in pickup location, traffic, or fare expectation.  
- 🙋 {strong('Customer cancellations')}: **{outcomes['Cancel_by Customer']:.1f}%** — often linked to wait time or price changes.  
- 🕳 {strong('Incomplete')}: **{outcomes['Incomplete']:.1f}%** — dropped mid-journey or technical issues.  
"""))

with st.expander("🔮 What could happen next?"):
    # Forecast next week's completion %
    wk = df.groupby("Week")["is_completed"].mean().mul(100)
    next_comp = trend_next_value(wk)
    if pd.notna(next_comp):
        st.info(f"Projected completion rate next period: **{next_comp:.1f}%** (simple trend).")
    else:
        st.info("Not enough history to project next period.")

st.markdown("---")


# 2) Hourly Demand
st.subheader("2️⃣ Demand by Hour of Day")
fig, ax = plt.subplots(figsize=(10,4))
sns.countplot(x="hour", data=df, color=ACCENT, ax=ax)
ax.set_title("Bookings by Hour")
ax.set_xlabel("Hour of Day")
st.pyplot(fig)

hour_counts = df["hour"].value_counts().sort_index()
if not hour_counts.empty:
    peak_hour = hour_counts.idxmax()
    st.markdown(f"- ⏰ Peak booking hour: **{peak_hour}:00**.")
else:
    st.markdown("- No hour data available in current selection.")

with st.expander("🔮 What could happen next?"):
    # Forecast next hour demand level using last 12h mean
    # (for filtered selection, interpret as relative peak expectation)
    next_hour_est = hour_counts.tail(12).mean() if len(hour_counts) >= 3 else hour_counts.mean()
    if pd.notna(next_hour_est):
        st.info(f"Expected bookings for the next peak hour: **~{int(round(next_hour_est))}** (very rough).")
    else:
        st.info("Not enough info to estimate next hour demand.")

st.markdown("---")


# 3) Weekday Demand
st.subheader("3️⃣ Demand by Weekday")
fig, ax = plt.subplots(figsize=(10,4))
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sns.countplot(x="day_of_week", data=df, order=order, palette=PALETTE, ax=ax)
ax.set_title("Bookings by Day of Week")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
st.pyplot(fig)

dow_counts = df["day_of_week"].value_counts()
if not dow_counts.empty:
    top_dow = dow_counts.idxmax()
    st.markdown(f"- 📅 Highest demand day: **{top_dow}**.")
with st.expander("🔮 What could happen next?"):
    st.info("If our filters lean toward commuters, expect weekday peaks; leisure filters lean toward weekend spikes.")

st.markdown("---")


# 4) Weekly Trends (Completion / Cancels)
st.subheader("4️⃣ Weekly Completion & Cancellation Trends")
weekly = (df
          .groupby("Week")[["is_completed","is_cancel_driver","is_cancel_customer"]]
          .mean().mul(100)
         )
weekly = weekly.rename(columns={
    "is_completed": "% Completed",
    "is_cancel_driver": "% Cancelled by Driver",
    "is_cancel_customer": "% Cancelled by Customer"
})

fig, ax = plt.subplots(figsize=(12,5))
weekly.plot(marker="o", ax=ax, color=[PALETTE[0], PALETTE[4], PALETTE[3]])
ax.set_title("Weekly Outcome Rates (%)")
ax.set_ylabel("Percent")
ax.grid(True, alpha=0.2)
st.pyplot(fig)

with st.expander("🔮 What could happen next?"):
    next_wk_completed = trend_next_value(weekly["% Completed"]) if "% Completed" in weekly else np.nan
    if pd.notna(next_wk_completed):
        st.info(f"Projected next week % Completed: **{next_wk_completed:.1f}%**.")
    else:
        st.info("Insufficient history to project next week outcomes.")

st.markdown("---")


# 5) Distance vs Completion
st.subheader("5️⃣ Distance vs Completion")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(
    x=df["is_completed"].map({0: "Not Completed", 1: "Completed"}),
    y="distance", data=df, palette=[PALETTE[4], PALETTE[1]], ax=ax
)
ax.set_title("Distance by Outcome")
ax.set_xlabel("Outcome")
ax.set_ylabel("Ride Distance (km)")
st.pyplot(fig)

comp_dist = df.loc[df["is_completed"]==1, "distance"].median()
noncomp_dist = df.loc[df["is_completed"]==0, "distance"].median()
st.markdown(f"- 📏 Median distance — {strong('Completed')}: **{comp_dist:.1f} km** | {strong('Not Completed')}: **{noncomp_dist:.1f} km**.")

with st.expander("🔮 What could happen next?"):
    st.info("If long trips are rising, consider incentives for drivers to lower cancellations on longer routes.")

st.markdown("---")


# 6) Cancellation Breakdown & Reasons
st.subheader("6️⃣ Cancellations: Who and Why?")
cancel_counts = {
    "Driver Cancel": int(df["is_cancel_driver"].sum()),
    "Customer Cancel": int(df["is_cancel_customer"].sum()),
    "Incomplete": int(df["is_incomplete"].sum())
}

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=list(cancel_counts.keys()), y=list(cancel_counts.values()), palette=[PALETTE[4], PALETTE[3], PALETTE[2]], ax=ax)
ax.set_title("Cancellation & Incomplete (Counts)")
ax.set_ylabel("Count")
for i, v in enumerate(cancel_counts.values()):
    ax.text(i, v + max(1, v*0.01), f"{v:,}", ha="center", va="bottom")
st.pyplot(fig)

# Top cancellation reasons if available
if "Driver Cancellation Reason" in df.columns:
    top_dr = top_k(df.loc[df["is_cancel_driver"]==1, "Driver Cancellation Reason"], k=3)
    if not top_dr.empty:
        st.markdown("**Top Driver Cancellation Reasons:**")
        for i, (rsn, cnt) in enumerate(top_dr.items(), 1):
            st.markdown(f"- {i}. {rsn} — **{cnt}**")
if "Reason for cancelling by Customer" in df.columns:
    top_cr = top_k(df.loc[df["is_cancel_customer"]==1, "Reason for cancelling by Customer"], k=3)
    if not top_cr.empty:
        st.markdown("**Top Customer Cancellation Reasons:**")
        for i, (rsn, cnt) in enumerate(top_cr.items(), 1):
            st.markdown(f"- {i}. {rsn} — **{cnt}**")

with st.expander("🔮 What could happen next?"):
    st.info("Reduce driver cancels with better fare transparency and pickup clustering; reduce customer cancels with tighter ETAs and surge communication.")

st.markdown("---")


# 7) Ratings (Driver & Customer)
st.subheader("7️⃣ Experience: Ratings")
fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.histplot(df["driver_rating"].dropna(), bins=10, kde=True, ax=ax[0], color=PALETTE[0])
ax[0].set_title("Driver Ratings")
sns.histplot(df["cust_rating"].dropna(), bins=10, kde=True, ax=ax[1], color=PALETTE[1])
ax[1].set_title("Customer Ratings")
st.pyplot(fig)

st.markdown(
    f"- ⭐ Avg Driver: **{avg_driver_rating:.2f}** | ⭐ Avg Customer: **{avg_cust_rating:.2f}**"
)

with st.expander("🔮 What could happen next?"):
    st.info("Monitor dips after peak hours or bad weather to preempt support escalations.")

st.markdown("---")


# 8) Vehicle Type Preference
st.subheader("8️⃣ Vehicle Type Preference")
fig, ax = plt.subplots(figsize=(8,5))
order = df["Vehicle Type"].value_counts().index
sns.countplot(y="Vehicle Type", data=df, order=order, palette=PALETTE, ax=ax)
ax.set_title("Most Booked Vehicle Types")
ax.set_xlabel("Count")
st.pyplot(fig)

top_veh = order[0] if len(order) else None
if top_veh:
    st.markdown(f"- 🚗 Most popular vehicle type: **{top_veh}** in the current selection.")

with st.expander("🔮 What could happen next?"):
    st.info("If demand for SUVs spikes in weekends/evenings, consider targeted onboarding or incentives.")

st.markdown("---")


# 9) Payment Method Share
st.subheader("9️⃣ Payment Method Share")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x="Payment Method", data=df, palette=PALETTE, ax=ax)
ax.set_title("Payment Preference")
ax.set_xlabel("")
st.pyplot(fig)

pm = df["Payment Method"].value_counts(normalize=True)
if not pm.empty:
    st.markdown(
        "- 💳 Current split: " + 
        " | ".join([f"**{m}**: {pct(v)}" for m, v in pm.items()])
    )

with st.expander("🔮 What could happen next?"):
    st.info("Expect digital payments to rise during surge times and office hours.")

st.markdown("---")


# 10) Revenue Trend + Projection
st.subheader("🔟 Daily Revenue Trend & Projection")
daily_rev = df.groupby("date_only")["revenue"].sum(min_count=1).sort_index()

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(daily_rev.index, daily_rev.values, marker="o", linewidth=1.8, color=PALETTE[0])
ax.set_title("Daily Revenue")
ax.set_ylabel("Revenue (₹)")
ax.grid(True, alpha=0.2)
st.pyplot(fig)

next_rev = trend_next_value(daily_rev.values)
if pd.notna(next_rev):
    st.markdown(f"- 💰 Projected next day revenue (simple trend): **₹{next_rev:,.0f}**.")

with st.expander("🔮 What could happen next?"):
    st.info("Revenue typically follows demand peaks (weekday commute, weekend evenings). Watch cancellations—each 1% increase can hurt daily revenue meaningfully.")

# Footer
st.success("All visuals and insights update instantly with our filters. Explore patterns and projections across time, distance, vehicles, and payments.")
