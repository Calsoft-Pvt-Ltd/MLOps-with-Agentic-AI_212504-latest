from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

# Generate Data
X, y = make_regression(n_samples=100, n_features=10, n_informative=7,
                       noise=0.1, random_state=42)

# Scale features to positive range [0, 100]
X = (X - X.min()) / (X.max() - X.min()) * 100

# Scale target to positive range
y = (y - y.min()) / (y.max() - y.min()) * 1000

# Create DataFrame and save to CSV
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df.to_csv('synthetic_regression_data.csv', index=False)
print(f"Data saved to 'synthetic_regression_data.csv'")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData ranges:")
print(f"Features: [{X.min():.2f}, {X.max():.2f}]")
print(f"Target: [{y.min():.2f}, {y.max():.2f}]")