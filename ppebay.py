#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import os
import uuid
import time
import torch
import joblib
import hashlib
import logging
import base64
import requests
import numpy as np
import pandas as pd

import streamlit as st

try:
    from clip_utils import get_clip_tags
    st.info("✅ CLIP module loaded")
except Exception as e:
    st.error(f"❌ Failed to load CLIP module: {e}")

# In[2]:


from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from transformers import CLIPProcessor, CLIPModel
from werkzeug.utils import secure_filename


# In[3]:


# GDPR Logging
logging.basicConfig(filename='pricepulse_gdpr.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("GDPR Notice: All personally identifiable information is pseudonymized or anonymized.")
logging.info("Data collection is limited to what's necessary for pricing optimization purposes.")
logging.info("Retention policy: Synthetic or anonymized data retained for training; real data auto-purged after 90 days.")


# In[4]:


# Flask App Setup
app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)
CORS(app)
np.random.seed(42)


# In[5]:


# CLIP Image Model Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

candidate_labels = ["electronics", "kitchen", "fitness", "gaming", "accessories", "smartphone", "furniture",
                    "toys", "clothing", "beauty", "books", "home decor", "headphones", "tools", "office supplies"]

def analyze_product_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().detach().numpy().flatten()
        tags = [candidate_labels[i] for i in probs.argsort()[-3:][::-1]]
        return {"tags": tags, "scores": [round(probs[i], 3) for i in probs.argsort()[-3:][::-1]]}
    except Exception as e:
        logging.error(f"Image processing error: {str(e)}")
        return {"tags": [], "error": str(e)}


# In[6]:


# Synthetic Dataset
def load_extended_dataset():
    num_samples = 1000
    df = pd.DataFrame({
        'base_price': np.random.uniform(10, 100, num_samples),
        'competitor_price': np.random.uniform(8, 110, num_samples),
        'demand': np.random.randint(50, 500, num_samples),
        'customer_rating': np.random.uniform(3, 5, num_samples),
        'inventory_level': np.random.randint(10, 100, num_samples)
    })
    df['optimal_price'] = df['base_price'] * 0.8 + df['competitor_price'] * 0.1 + df['customer_rating'] * 0.1
    df['seller_id'] = [uuid.uuid4().hex for _ in range(len(df))]
    df['product_id'] = [hashlib.sha256(str(x).encode()).hexdigest()[:12] for x in df.index]
    df['seller_rating'] = np.random.uniform(2.5, 5.0, len(df))
    df['location'] = ['SE', 'UK', 'DE', 'FR', 'NL'] * (len(df) // 5)
    df['brand'] = np.random.choice(['BrandA', 'BrandB', 'BrandC'], len(df))
    df['weight_grams'] = np.random.uniform(100, 1500, len(df))
    df['sales_last_30d'] = np.random.randint(5, 300, len(df))
    df['revenue'] = df['sales_last_30d'] * df['base_price']
    df['ad_spend'] = np.random.uniform(5, 50, len(df))
    df['ctr'] = np.random.uniform(0.01, 0.3, len(df))
    df['customer_comments'] = np.random.choice(['Great', 'Okay', 'Bad'], len(df))
    df['returns'] = np.random.randint(0, 10, len(df))
    df['warehouse'] = np.random.choice(['WH1', 'WH2', 'WH3'], len(df))
    df['shipping_delay_days'] = np.random.randint(0, 10, len(df))
    df['seasonality_factor'] = np.random.uniform(0.8, 1.2, len(df))
    df['launch_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    return df


# In[7]:


def preprocess_data(df):
    df['price_difference'] = df['base_price'] - df['competitor_price']
    features = [
        'price_difference', 'demand', 'customer_rating', 'inventory_level',
        'seller_rating', 'sales_last_30d', 'revenue', 'ad_spend',
        'ctr', 'returns', 'shipping_delay_days', 'seasonality_factor'
    ]
    df = df.fillna(df.median(numeric_only=True))
    return df, features


# In[8]:


# Model Training or Loading
df = load_extended_dataset()
df, features = preprocess_data(df)
X = df[features]
y = df['optimal_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_path = 'rf_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)


# In[9]:


# eBay Sandbox Credentials (Replace with your credentials)
EBAY_CLIENT_ID = "YOUR_SANDBOX_CLIENT_ID"
EBAY_CLIENT_SECRET = "YOUR_SANDBOX_CLIENT_SECRET"
EBAY_SCOPE = "https://api.ebay.com/oauth/api_scope"
EBAY_OAUTH_ENDPOINT = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"


# In[10]:


def get_ebay_access_token():
    try:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + base64.b64encode(f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode()).decode()
        }
        data = {
            "grant_type": "client_credentials",
            "scope": EBAY_SCOPE
        }
        response = requests.post(EBAY_OAUTH_ENDPOINT, headers=headers, data=data)
        response.raise_for_status()
        token = response.json()['access_token']
        return token
    except Exception as e:
        logging.error(f"eBay OAuth Error: {str(e)}")
        return None


# In[11]:


@app.route('/predict', methods=['POST'])
@limiter.limit('10 per minute')
def predict_optimal_price():
    try:
        data = request.get_json()
        required_fields = set(features)
        if not required_fields.issubset(data.keys()):
            return jsonify({'error': 'Missing required fields.'}), 400
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data[list(required_fields)])[0]
        logging.info(f"Prediction hash: {hashlib.md5(str(data).encode()).hexdigest()} → {prediction}")
        return jsonify({'optimal_price': round(prediction, 2)})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400


# In[12]:


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        file.save(filepath)
        result = analyze_product_image(filepath)
        os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# In[13]:


@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features': features})


# In[14]:


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running'})


# In[15]:


@app.route('/sandbox_test', methods=['GET'])
def ebay_sandbox_test():
    try:
        token = get_ebay_access_token()
        if not token:
            return jsonify({'error': 'Failed to authenticate with eBay'}), 500
        mock_product = {
            'price_difference': 5.0,
            'demand': 200,
            'customer_rating': 4.5,
            'inventory_level': 30,
            'seller_rating': 4.7,
            'sales_last_30d': 150,
            'revenue': 1500.0,
            'ad_spend': 25.0,
            'ctr': 0.15,
            'returns': 2,
            'shipping_delay_days': 1,
            'seasonality_factor': 1.0
        }
        prediction = model.predict(pd.DataFrame([mock_product]))[0]
        return jsonify({
            'input': mock_product,
            'predicted_optimal_price': round(prediction, 2)
        })
    except Exception as e:
        logging.error(f"Sandbox Test Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
# Force save the trained model even if it already exists
joblib.dump(model, "rf_model.joblib")
print("✅ Model saved as rf_model.joblib")

if __name__ == '__main__':
    print("GDPR-compliant PricePulse API with eBay Sandbox is running...")
    app.run(host='127.0.0.1', port=5000, debug=True)


# In[ ]:





# In[ ]:




