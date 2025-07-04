
AI-Powered Pricing Optimizer for eCommerce Sellers

This project is an AI-driven pricing assistant designed to help online sellers across all major eCommerce platforms — including **eBay, Amazon, Etsy**, and independent storefronts — find the optimal price point for their products.

Features

-Streamlit web app for easy input and instant price recommendations
- Random Forest regression model trained on synthetic eCommerce data
- Takes into account factors like demand, customer ratings, returns, ad spend, seasonality, and more
- GDPR-friendly logging, with anonymized data and auto-purge policy
- Optional: CLIP-based product image classification
- Optional: eBay sandbox API test integration

Use Cases

- Solo sellers and small brands looking to optimize pricing
- AI product developers testing pricing strategy tools
- Marketplace analytics experiments

Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- joblib
- (Optional) HuggingFace Transformers for image tagging

Installation

Clone the repo:

bash
git clone https://github.com/Mannamu/ai-pricing-optimizer.git
cd ai-pricing-optimizer


Install dependencies:
bash
pip install -r requirements.txt


Ensure the trained model file `rf_model.joblib` is in the root directory. Then run the app:

bash
streamlit run app.py
```

Sample Inputs

The model expects inputs such as:

- price_difference: Base price minus competitor price
- demand: Search/popularity score (0–500)
- customer_rating: Average customer rating (1.0–5.0)
- sales_last_30d, revenue, ctr,ad_spend,returns, etc.

Deployment

You can deploy your app for free on:

- Streamlit Community Cloud](https://streamlit.io/cloud)
- Hugging Face Spaces](https://huggingface.co/spaces)
- Replit(https://replit.com)

Contributing

This tool is in beta and we're collecting feedback from real sellers. Feel free to fork, test, or raise issues!



Made with ❤️ for the future of fair, data-driven pricing.

