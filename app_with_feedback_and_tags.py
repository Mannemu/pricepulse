import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
from datetime import datetime

# Load model
model = joblib.load("rf_model.joblib")

# Set page config
st.set_page_config(page_title="AI Price Optimizer", layout="centered")

# Session state for logs
if "log" not in st.session_state:
    st.session_state.log = []

# Sidebar for navigation
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", ["📦 Price Prediction", "🧪 Feedback Form", "📊 Dev Dashboard"])

# --- Helper for image tagging (simulated tags for now) ---
def fake_clip_tagging(image):
    return ["electronics", "accessories", "gadgets"]

# --- Page 1: Price Prediction ---
if view == "📦 Price Prediction":
    st.title("📦 AI-Powered Price Optimizer")
    st.markdown("Upload a product image (optional) and fill out product signals for a price suggestion.")

    uploaded_image = st.file_uploader("Upload product image (optional)", type=["jpg", "jpeg", "png"])
    tags = []
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Simulated tag output
        tags = fake_clip_tagging(image)
        st.info(f"Predicted Tags: {', '.join(tags)}")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            price_diff = st.number_input("Price Difference (Base - Competitor)", value=5.0)
            demand = st.slider("Demand (search/popularity)", 0, 500, 200)
            rating = st.slider("Customer Rating", 1.0, 5.0, 4.5)
            inventory = st.number_input("Inventory Level", value=30)
            seller_rating = st.slider("Seller Rating", 1.0, 5.0, 4.8)
        with col2:
            sales = st.number_input("Sales Last 30 Days", value=150)
            revenue = st.number_input("Revenue (USD)", value=1500.0)
            ad_spend = st.number_input("Ad Spend (USD)", value=25.0)
            ctr = st.slider("Click-Through Rate (CTR)", 0.0, 0.5, 0.15)
            returns = st.slider("Returns (Last 30 Days)", 0, 20, 2)
            delay = st.slider("Shipping Delay (Days)", 0, 10, 1)
            seasonality = st.slider("Seasonality Factor", 0.5, 1.5, 1.0)

        submitted = st.form_submit_button("Get My Price Prediction")
        if submitted:
            input_data = pd.DataFrame([{
                'price_difference': price_diff,
                'demand': demand,
                'customer_rating': rating,
                'inventory_level': inventory,
                'seller_rating': seller_rating,
                'sales_last_30d': sales,
                'revenue': revenue,
                'ad_spend': ad_spend,
                'ctr': ctr,
                'returns': returns,
                'shipping_delay_days': delay,
                'seasonality_factor': seasonality
            }])

            prediction = model.predict(input_data)[0]
            st.success(f"💡 Recommended Optimal Price: **${round(prediction, 2)}**")

            # Log entry
            st.session_state.log.append({
                "timestamp": datetime.now().isoformat(),
                "price": round(prediction, 2),
                "tags": tags,
                "input": input_data.to_dict(orient="records")[0]
            })

# --- Page 2: Feedback Form ---
elif view == "🧪 Feedback Form":
    st.title("🧪 Beta Tester Feedback")
    st.markdown("We’d love your feedback on how useful the prediction was!")

    with st.form("feedback_form"):
        accuracy = st.radio("Was the price prediction accurate?", ["Yes", "No", "Somewhat"])
        comment = st.text_area("Any suggestions or feedback?")
        email = st.text_input("Your email (optional)")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            st.success("✅ Thank you for your feedback!")
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "comment": comment,
                "email": email
            }
            if "feedback" not in st.session_state:
                st.session_state.feedback = []
            st.session_state.feedback.append(feedback)

# --- Page 3: Dev Dashboard ---
elif view == "📊 Dev Dashboard":
    st.title("📊 Developer Dashboard")

    logs = st.session_state.get("log", [])
    feedbacks = st.session_state.get("feedback", [])

    st.subheader("Recent Predictions")
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs[["timestamp", "price", "tags"]])
        st.metric("🔢 Total Predictions", len(df_logs))
        st.metric("💰 Average Price", f"${df_logs['price'].mean():.2f}")
    else:
        st.info("No predictions yet.")

    st.subheader("User Feedback")
    if feedbacks:
        df_feedback = pd.DataFrame(feedbacks)
        st.dataframe(df_feedback)
    else:
        st.info("No feedback submitted yet.")
