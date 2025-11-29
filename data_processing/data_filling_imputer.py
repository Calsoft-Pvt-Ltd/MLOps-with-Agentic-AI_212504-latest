# using sklearn imputer
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Sample dataset
data = pd.read_csv('data_with_faults.csv')
df = pd.DataFrame(data)

# original data
print(f"\nMissing values:\n{df.isna().sum()}")

# Remove duplicates
df = df.drop_duplicates()
print(f"\nAfter removing duplicates: {df.shape}")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")
#df[['column1','column2']]
# Handle numeric columns with SimpleImputer (median strategy)
if numeric_cols:
    imp_numeric = SimpleImputer(missing_values=np.nan, strategy='median')
    df[numeric_cols] = imp_numeric.fit_transform(df[numeric_cols])
    print("\nNumeric columns filled with median")

# Handle categorical columns with SimpleImputer (most_frequent strategy)
if categorical_cols:
    imp_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df[categorical_cols] = imp_categorical.fit_transform(df[categorical_cols])
    print("Categorical columns filled with most frequent value (mode)")

print(f"\nMissing values after cleaning:\n{df.isna().sum()}")