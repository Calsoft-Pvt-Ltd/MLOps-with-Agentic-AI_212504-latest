from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

# Generate Data with positive values directly
np.random.seed(42)

# Generate positive features using random distributions
n_samples = 100
n_features = 10
n_informative = 7

# Create positive features (e.g., exponential or uniform distributions)
X = np.random.exponential(scale=50, size=(n_samples, n_features))

# Generate target as linear combination of features + noise
true_coef = np.zeros(n_features)
true_coef[:n_informative] = np.random.uniform(5, 20, n_informative)
y = X @ true_coef + np.random.normal(0, 10, n_samples)

# Ensure target is positive
y = np.abs(y)

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