import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
"""
KNN Imputation:
  ✓ Fast and simple
  ✓ Uses nearest neighbors based on feature similarity
  ✓ Good for datasets with clear patterns
  ✗ Sensitive to outliers
  ✗ Doesn't account for relationships between features well

MICE (Iterative) Imputation: Multivariate Imputation by Chained Equation
  ✓ More sophisticated - models each feature
  ✓ Captures complex relationships between variables
  ✓ Better for datasets with strong feature correlations
  ✗ Slower (computationally intensive)
  ✗ May overfit on small datasets

Recommendation:
  - Use KNN for quick imputation with moderate accuracy
  - Use MICE for higher accuracy when you have enough data and time
"""
# Sample dataset
data = pd.read_csv('data_with_faults.csv')
df = pd.DataFrame(data)

print("="*60)
print("ADVANCED IMPUTATION METHODS")
print("="*60)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nMissing values before imputation:")
print(df.isna().sum())

# Remove duplicates first
df = df.drop_duplicates()
print(f"\nAfter removing duplicates: {df.shape}")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# ========================================
# METHOD 1: KNN IMPUTER
# ========================================
print("\n" + "="*60)
print("METHOD 1: KNN IMPUTATION")
print("="*60)

# Create a copy for KNN imputation
df_knn = df.copy()

# Handle categorical columns first (use mode for categorical)
if categorical_cols:
    for col in categorical_cols:
        mode_value = df_knn[col].mode()[0] if not df_knn[col].mode().empty else 'Unknown'
        df_knn[col].fillna(mode_value, inplace=True)
    print(f"✓ Categorical columns filled with mode: {categorical_cols}")

# Apply KNN Imputer on numeric columns
if numeric_cols:
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_knn[numeric_cols] = knn_imputer.fit_transform(df_knn[numeric_cols])
    print(f"✓ KNN Imputation completed on numeric columns")
    print(f"  - Number of neighbors: 5")
    print(f"  - Weight: distance (closer neighbors have more influence)")

print(f"\nMissing values after KNN imputation:")
print(df_knn.isna().sum())

# Save KNN imputed data
df_knn.to_csv('data_knn_imputed.csv', index=False)
print(f"\n✓ Saved to 'data_knn_imputed.csv'")

# ========================================
# METHOD 2: MICE (Iterative) IMPUTER
# ========================================
print("\n" + "="*60)
print("METHOD 2: MICE (Multiple Imputation by Chained Equations)")
print("="*60)

# Create a copy for MICE imputation
df_mice = df.copy()

# Handle categorical columns first
if categorical_cols:
    for col in categorical_cols:
        mode_value = df_mice[col].mode()[0] if not df_mice[col].mode().empty else 'Unknown'
        df_mice[col].fillna(mode_value, inplace=True)
    print(f"✓ Categorical columns filled with mode: {categorical_cols}")

# Apply MICE Imputer on numeric columns
if numeric_cols:
    # Using RandomForestRegressor as the estimator for better performance
    mice_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42),
        max_iter=10,
        random_state=42,
        verbose=0
    )
    df_mice[numeric_cols] = mice_imputer.fit_transform(df_mice[numeric_cols])
    print(f"✓ MICE Imputation completed on numeric columns")
    print(f"  - Estimator: RandomForestRegressor")
    print(f"  - Max iterations: 10")
    print(f"  - Each feature is modeled as a function of other features")

print(f"\nMissing values after MICE imputation:")
print(df_mice.isna().sum())

# Save MICE imputed data
df_mice.to_csv('data_mice_imputed.csv', index=False)
print(f"\n✓ Saved to 'data_mice_imputed.csv'")

# ========================================
# COMPARISON
# ========================================
print("\n" + "="*60)
print("COMPARISON OF METHODS")
print("="*60)

# Compare a sample of imputed values for a specific column
if 'math_score' in numeric_cols:
    # Find rows that originally had missing math_score
    missing_indices = df[df['math_score'].isna()].index[:5]  # First 5 missing values
    
    if len(missing_indices) > 0:
        print("\nSample comparison (first 5 originally missing math_score values):")
        print(f"{'Index':<10} {'KNN Imputed':<15} {'MICE Imputed':<15}")
        print("-" * 40)
        for idx in missing_indices:
            knn_val = df_knn.loc[idx, 'math_score']
            mice_val = df_mice.loc[idx, 'math_score']
            print(f"{idx:<10} {knn_val:<15.2f} {mice_val:<15.2f}")

# Statistical comparison
print("\nStatistical summary of imputed data:")
print("\nOriginal data (excluding NaN):")
print(df[numeric_cols].describe())

print("\nKNN Imputed data:")
print(df_knn[numeric_cols].describe())

print("\nMICE Imputed data:")
print(df_mice[numeric_cols].describe())
