# Copyright (c) 2025 Manna Mulanga. All Rights Reserved.
# Helper functions to communicate with eBay Sandbox API (Browse endpoint).

import requests
import streamlit as st

EBAY_API_ENDPOINT = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"

def search_ebay_sandbox(query, limit=5):
    """
    Search for items on the eBay Sandbox based on a keyword query.

    Args:
        query (str): The search query (e.g., product name).
        limit (int): Maximum number of results to return.

    Returns:
        list: A list of dicts with item details (title, price, currency, condition).
    """
    headers = {
        "Authorization": f"Bearer {st.secrets['ebay']['access_token']}",
        "Content-Type": "application/json",
    }
    params = {
        "q": query,
        "limit": limit
    }

    response = requests.get(EBAY_API_ENDPOINT, headers=headers, params=params)

    if response.status_code != 200:
        st.error(f"eBay API error: {response.status_code} {response.text}")
        return []

    data = response.json()
    return [
        {
            "title": item.get("title"),
            "price": float(item["price"]["value"]),
            "currency": item["price"]["currency"],
            "condition": item.get("condition", "N/A")
        }
        for item in data.get("itemSummaries", [])
    ]
