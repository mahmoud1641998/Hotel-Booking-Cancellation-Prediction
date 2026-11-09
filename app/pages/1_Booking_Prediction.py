
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Add Project Root to Python Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_booking, predict_booking_probability


# Page Configuration
st.set_page_config(
    page_title="Booking Prediction",
    page_icon="🔮",
    layout="wide"
)


# Load Model
@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# Page Header
st.title("🔮 Hotel Booking Prediction")

st.write(
    "Enter the booking details below to predict "
    "whether the booking is likely to be canceled."
)

st.divider()


# Booking Information Form
with st.form("booking_prediction_form"):

    st.header("Booking Information")

    col1, col2 = st.columns(2)

    with col1:
        adults = st.number_input(
            "Number of Adults",
            min_value=0,
            value=2,
            step=1
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            value=0,
            step=1
        )

        weekend_nights = st.number_input(
            "Number of Weekend Nights",
            min_value=0,
            value=1,
            step=1
        )

        week_nights = st.number_input(
            "Number of Week Nights",
            min_value=0,
            value=2,
            step=1
        )

        parking = st.selectbox(
            "Car Parking Space",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        lead_time = st.number_input(
            "Lead Time",
            min_value=0,
            value=30,
            step=1
        )

        repeated = st.selectbox(
            "Repeated Guest",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        previous_cancellations = st.number_input(
            "Previous Cancellations (P-C)",
            min_value=0,
            value=0,
            step=1
        )

        previous_non_cancellations = st.number_input(
            "Previous Non-Cancellations (P-not-C)",
            min_value=0,
            value=0,
            step=1
        )

        average_price = st.number_input(
            "Average Price",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        special_requests = st.number_input(
            "Special Requests",
            min_value=0,
            value=0,
            step=1
        )

        meal_type = st.selectbox(
            "Type of Meal",
            options=[
                "Meal Plan 1",
                "Not Selected",
                "Meal Plan 2",
                "Meal Plan 3"
            ]
        )

        room_type = st.selectbox(
            "Room Type",
            options=[
                "Room_Type 1",
                "Room_Type 2",
                "Room_Type 3",
                "Room_Type 4",
                "Room_Type 5",
                "Room_Type 6",
                "Room_Type 7"
            ]
        )

        market_segment = st.selectbox(
            "Market Segment Type",
            options=[
                "Online",
                "Offline",
                "Corporate",
                "Complementary",
                "Aviation"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Booking Status",
        use_container_width=True
    )


# Prediction
if submitted:

    if adults == 0 and children == 0:
        st.error(
            "A booking must include at least one adult or child."
        )

    else:
        input_data = pd.DataFrame([{
            "number of adults": adults,
            "number of children": children,
            "number of weekend nights": weekend_nights,
            "number of week nights": week_nights,
            "car parking space": parking,
            "lead time": lead_time,
            "repeated": repeated,
            "P-C": previous_cancellations,
            "P-not-C": previous_non_cancellations,
            "average price": average_price,
            "special requests": special_requests,
            "type of meal": meal_type,
            "room type": room_type,
            "market segment type": market_segment
        }])

        prediction = predict_booking(input_data, model)[0]

        probabilities = predict_booking_probability(
            input_data,
            model
        )

        canceled_probability = probabilities.loc[
            probabilities.index[0], "Canceled"
        ]

        not_canceled_probability = probabilities.loc[
            probabilities.index[0], "Not_Canceled"
        ]

        st.divider()
        st.header("Prediction Result")

        if prediction == "Canceled":
            st.error("Prediction: Canceled")

        else:
            st.success("Prediction: Not Canceled")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Cancellation Probability",
                f"{canceled_probability:.2%}"
            )

        with col2:
            st.metric(
                "Non-Cancellation Probability",
                f"{not_canceled_probability:.2%}"
            )

        st.subheader("Prediction Probabilities")

        probability_chart = pd.DataFrame(
            {
                "Probability": [
                    canceled_probability,
                    not_canceled_probability
                ]
            },
            index=["Canceled", "Not_Canceled"]
        )

        st.bar_chart(probability_chart)
