import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Create synthetic dataset
num_samples = 1000
df = pd.DataFrame({
    'base_price': np.random.uniform(10, 100, num_samples),
    'competitor_price': np.random.uniform(8, 110, num_samples),
    'demand': np.random.randint(50, 500, num_samples),
    'customer_rating': np.random.uniform(3, 5, num_samples),
    'inventory_level': np.random.randint(10, 100, num_samples),
    'seller_rating': np.random.uniform(2.5, 5.0, num_samples),
    'sales_last_30d': np.random.randint(5, 300, num_samples),
    'revenue': np.random.uniform(500, 10000, num_samples),
    'ad_spend': np.random.uniform(5, 50, num_samples),
    'ctr': np.random.uniform(0.01, 0.3, num_samples),
    'returns': np.random.randint(0, 10, num_samples),
    'shipping_delay_days': np.random.randint(0, 10, num_samples),
    'seasonality_factor': np.random.uniform(0.8, 1.2, num_samples)
})
df['price_difference'] = df['base_price'] - df['competitor_price']
df['optimal_price'] = df['base_price'] * 0.8 + df['competitor_price'] * 0.1 + df['customer_rating'] * 0.1

# Prepare features and target
features = [
    'price_difference', 'demand', 'customer_rating', 'inventory_level',
    'seller_rating', 'sales_last_30d', 'revenue', 'ad_spend',
    'ctr', 'returns', 'shipping_delay_days', 'seasonality_factor'
]
X = df[features]
y = df['optimal_price']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.joblib")
print("✅ Model trained and saved to rf_model.joblib")
