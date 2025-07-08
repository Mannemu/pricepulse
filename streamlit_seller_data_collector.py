# streamlit_seller_data_collector.py
# Streamlit Frontend for Seller Data Collection & Research

import streamlit as st
import pandas as pd
import requests
import os
import webbrowser
import urllib.parse
from requests.auth import HTTPBasicAuth

st.set_page_config(page_title="PricePulse Research Portal", layout="wide")
st.title("📊 Seller Data Collector – PricePulse")

st.markdown("""
This app collects anonymized seller data for research. You can:
- 📥 Submit product details
- 🖼 Upload a product image for tagging
- 🧠 Receive price predictions
- ✅ Your submissions help train better pricing models.
""")

st.subheader("1. Submit Product Details for Pricing")

with st.form("product_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        base_price = st.number_input("Base Price ($)", min_value=1.0, value=20.0)
        customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.5)
        seller_rating = st.slider("Seller Rating", 1.0, 5.0, 4.8)
        sales_last_30d = st.number_input("Sales (last 30d)", min_value=0, value=100)
    with col2:
        competitor_price = st.number_input("Competitor Price ($)", min_value=1.0, value=22.0)
        inventory_level = st.number_input("Inventory Level", min_value=0, value=50)
        revenue = base_price * sales_last_30d
        ad_spend = st.number_input("Ad Spend ($)", min_value=0.0, value=20.0)
    with col3:
        ctr = st.slider("Click-Through Rate (CTR)", 0.01, 0.5, 0.1)
        returns = st.number_input("Returns", min_value=0, value=3)
        shipping_delay_days = st.number_input("Shipping Delay (days)", min_value=0, value=2)
        seasonality_factor = st.slider("Seasonality Factor", 0.5, 2.0, 1.0)

    submit = st.form_submit_button("🔍 Predict Optimal Price")

    if submit:
        payload = {
            "price_difference": base_price - competitor_price,
            "demand": sales_last_30d * 2,
            "customer_rating": customer_rating,
            "inventory_level": inventory_level,
            "seller_rating": seller_rating,
            "sales_last_30d": sales_last_30d,
            "revenue": revenue,
            "ad_spend": ad_spend,
            "ctr": ctr,
            "returns": returns,
            "shipping_delay_days": shipping_delay_days,
            "seasonality_factor": seasonality_factor
        }

        # Save to CSV
        df = pd.DataFrame([payload])
        df.to_csv("submissions.csv", mode='a', index=False, header=not os.path.exists("submissions.csv"))

        try:
            res = requests.post("https://pricepulse-dvwzd9xn28abtfytaarvcz.streamlit.app/predict", json=payload)
            result = res.json()
            if "optimal_price" in result:
                st.success(f"💰 Predicted Optimal Price: ${result['optimal_price']:.2f}")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"API error: {e}")

# 2. Upload image for auto-tagging
st.subheader("2. Upload Product Image for Category Tagging")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    with st.spinner("Analyzing image..."):
        try:
            res = requests.post("https://pricepulse-dvwzd9xn28abtfytaarvcz.streamlit.app/analyze_image", files={"image": image_file})
            result = res.json()
            if "tags" in result:
                st.success(f"🏷️ Top Tags: {', '.join(result['tags'])}")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Image API error: {e}")

# 3. Test eBay Sandbox Endpoint
st.subheader("3. eBay Sandbox Test")
if st.button("🚀 Run eBay Sandbox Prediction Test"):
    try:
        res = requests.get("https://pricepulse-dvwzd9xn28abtfytaarvcz.streamlit.app/sandbox_test")
        result = res.json()
        if "predicted_optimal_price" in result:
            st.success(f"✅ Test Success! Predicted Price: ${result['predicted_optimal_price']:.2f}")
        else:
            st.warning(f"⚠️ {result.get('error', 'No prediction returned')}")
    except Exception as e:
        st.error(f"API error: {e}")

# 4. Connect to eBay Sandbox via OAuth2
st.subheader("4. Connect to eBay Sandbox via OAuth")
client_id = st.secrets["EBAY_CLIENT_ID"]
client_secret = st.secrets["EBAY_CLIENT_SECRET"]
redirect_uri = st.secrets["REDIRECT_URI"]

scopes = ["https://api.ebay.com/oauth/api_scope"]
params = {
    "client_id": client_id,
    "response_type": "code",
    "redirect_uri": redirect_uri,
    "scope": " ".join(scopes),
}

auth_url = f"https://auth.sandbox.ebay.com/oauth2/authorize?{urllib.parse.urlencode(params)}"
if st.button("🔐 Authorize with eBay"):
    st.markdown(f"[Click here to authenticate with eBay Sandbox]({auth_url})", unsafe_allow_html=True)

auth_code = st.text_input("Paste eBay Authorization Code here:")
if st.button("🚀 Exchange Code for Token") and auth_code:
    try:
        res = requests.post(
            "https://api.sandbox.ebay.com/identity/v1/oauth2/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri,
            },
            auth=HTTPBasicAuth(client_id, client_secret)
        )
        token_info = res.json()
        st.success("🎉 Token received!")
        st.json(token_info)
    except Exception as e:
        st.error(f"Token exchange failed: {e}")

# 5. Admin tools
st.subheader("5. Admin Tools")
if os.path.exists("submissions.csv"):
    st.download_button("⬇️ Download Submitted Data (CSV)", data=open("submissions.csv", "rb"), file_name="submissions.csv")
else:
    st.info("No submission data yet.")
