import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("rf_model.joblib")

FEATURES = [
    'price_difference', 'demand', 'customer_rating', 'inventory_level',
    'seller_rating', 'sales_last_30d', 'revenue', 'ad_spend',
    'ctr', 'returns', 'shipping_delay_days', 'seasonality_factor'
]

st.set_page_config(page_title="AI eBay Price Optimizer", layout="centered")
st.title("📦 AI-Powered Price Optimizer for eBay Sellers")
st.markdown("""
Use this tool to get a pricing recommendation for your eBay listing based on key product and market signals.
""")

with st.form("prediction_form"):
    price_diff = st.number_input("Price Difference (Base - Competitor)", value=5.0)
    demand = st.slider("Demand (search/popularity)", 0, 500, 200)
    rating = st.slider("Customer Rating", 1.0, 5.0, 4.5)
    inventory = st.number_input("Inventory Level", value=30)
    seller_rating = st.slider("Seller Rating", 1.0, 5.0, 4.8)
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