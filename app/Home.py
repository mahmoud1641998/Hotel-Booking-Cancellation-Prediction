
import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)


# Page Header
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write(
    "An End-to-End Machine Learning Application "
    "for Predicting Hotel Booking Cancellations."
)

st.divider()


# Project Overview
st.header("Project Overview")

st.write(
    """
    This application uses a Machine Learning model to predict
    whether a hotel booking is likely to be canceled or not.

    The system analyzes booking information such as lead time,
    number of guests, room type, market segment, average price,
    and other booking-related features.
    """
)


# Project Objective
st.header("Project Objective")

st.write(
    """
    The main objective is to help hotels better understand
    booking cancellation patterns and make more informed
    operational decisions.
    """
)


# Model Performance
st.header("Final Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Accuracy", "89.31%")
col2.metric("Canceled Precision", "87.43%")
col3.metric("Canceled Recall", "78.68%")
col4.metric("Canceled F1-Score", "82.82%")


st.caption(
    "Performance metrics are based on the held-out test dataset."
)

st.divider()


# Project Workflow
st.header("Machine Learning Workflow")

st.markdown(
    """
    1. **Data Understanding:** Explore and analyze the booking dataset.
    2. **Data Preprocessing:** Prepare numerical and categorical features.
    3. **Model Building:** Train and evaluate multiple classification models.
    4. **Hyperparameter Tuning:** Optimize the Random Forest model.
    5. **Prediction:** Use the final trained model to predict booking status.
    """
)


# Application Pages
st.header("Application Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Booking Prediction")
    st.write(
        "Enter booking details and get a prediction "
        "of whether the booking will be canceled."
    )

with col2:
    st.subheader("Data Insights")
    st.write(
        "Explore dataset statistics, booking patterns, "
        "and cancellation insights."
    )


st.divider()

st.caption(
    "Hotel Booking Cancellation Prediction System | "
    "Machine Learning Project"
)
