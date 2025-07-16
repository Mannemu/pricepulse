
import base64
import requests
import time
from pathlib import Path
import streamlit as st

SECRETS_PATH = Path(".streamlit/secrets.toml")
EXPIRY_LOG_PATH = Path(".streamlit/token_expiry.log")

def ensure_fresh_token():
    """Check and refresh token if needed — runs silently at app startup."""
    if is_token_expired():
        token = refresh_token()
        if token:
            update_secrets(token)

def is_token_expired():
    try:
        expiry_timestamp = int(EXPIRY_LOG_PATH.read_text())
        return time.time() >= expiry_timestamp - 60
    except FileNotFoundError:
        return True

def refresh_token():
    client_id = st.secrets["ebay"]["client_id"]
    client_secret = st.secrets["ebay"]["client_secret"]
    token_url = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {basic_auth}"
    }
    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }

    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]
        expires_in = int(token_data["expires_in"])
        expiry_timestamp = int(time.time()) + expires_in

        EXPIRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPIRY_LOG_PATH.write_text(str(expiry_timestamp))

        return access_token
    else:
        st.warning(f"❌ Token refresh failed: {response.text}")
        return None

def update_secrets(access_token):
    client_id = st.secrets["ebay"]["client_id"]
    client_secret = st.secrets["ebay"]["client_secret"]

    secrets_content = f"""
[ebay]
client_id = "{client_id}"
client_secret = "{client_secret}"
access_token = "{access_token}"
""".strip()

    SECRETS_PATH.write_text(secrets_content)
