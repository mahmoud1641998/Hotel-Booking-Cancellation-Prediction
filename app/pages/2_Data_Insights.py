
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Add Project Root to Python Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data


# Page Configuration
st.set_page_config(
    page_title="Data Insights",
    page_icon="📊",
    layout="wide"
)


# Load Dataset
@st.cache_data
def get_data():
    return load_data()


df = get_data()


# Page Header
st.title("📊 Hotel Booking Data Insights")

st.write(
    "Explore booking statistics and discover patterns "
    "related to hotel booking cancellations."
)

st.divider()


# Dataset Overview
st.header("Dataset Overview")

total_bookings = len(df)

canceled_bookings = (
    df["booking status"] == "Canceled"
).sum()

not_canceled_bookings = (
    df["booking status"] == "Not_Canceled"
).sum()

cancellation_rate = (
    canceled_bookings / total_bookings
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Canceled Bookings", f"{canceled_bookings:,}")
col3.metric("Not Canceled", f"{not_canceled_bookings:,}")
col4.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

st.divider()


# Booking Status Distribution
st.header("Booking Status Distribution")

status_counts = (
    df["booking status"]
    .value_counts()
    .reindex(["Canceled", "Not_Canceled"], fill_value=0)
)

st.bar_chart(status_counts)

st.divider()


# Cancellation Rate by Market Segment
st.header("Cancellation Rate by Market Segment")

market_cancellation = (
    df.groupby("market segment type")["booking status"]
    .apply(lambda x: (x == "Canceled").mean() * 100)
    .sort_values(ascending=False)
)

market_cancellation_df = market_cancellation.to_frame(
    name="Cancellation Rate (%)"
)

st.bar_chart(market_cancellation_df)

st.divider()


# Cancellation Rate by Room Type
st.header("Cancellation Rate by Room Type")

room_cancellation = (
    df.groupby("room type")["booking status"]
    .apply(lambda x: (x == "Canceled").mean() * 100)
    .sort_values(ascending=False)
)

room_cancellation_df = room_cancellation.to_frame(
    name="Cancellation Rate (%)"
)

st.bar_chart(room_cancellation_df)

st.divider()


# Average Lead Time by Booking Status
st.header("Average Lead Time by Booking Status")

lead_time_analysis = (
    df.groupby("booking status")["lead time"]
    .mean()
    .reindex(["Canceled", "Not_Canceled"])
)

lead_time_df = lead_time_analysis.to_frame(
    name="Average Lead Time"
)

st.bar_chart(lead_time_df)

st.divider()


st.caption(
    "Hotel Booking Cancellation Prediction System | "
    "Data Insights"
)
