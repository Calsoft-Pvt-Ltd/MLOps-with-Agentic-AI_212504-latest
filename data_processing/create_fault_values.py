import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the original CSV
df = pd.read_csv('data.csv')

# Create a copy
df_modified = df.copy()

# CONFIGURATION: Specify columns that should have outliers
OUTLIER_COLUMNS = ['math_score', 'reading_score', 'writing_score']
OUTLIERS_PER_COLUMN = 15  # Number of outliers per column

print("="*60)
print("CREATING DATASET WITH MISSING VALUES, DUPLICATES & OUTLIERS")
print("="*60)

# 1. ADD MISSING VALUES (20% of data)
print("\n1. Adding Missing Values...")
total_cells = df_modified.shape[0] * df_modified.shape[1]
missing_count = int(total_cells * 0.20)

# Get random positions for missing values
rows = np.random.choice(df_modified.shape[0], missing_count, replace=True)
cols = np.random.choice(df_modified.shape[1], missing_count, replace=True)

# Insert missing values at random positions
for row, col in zip(rows, cols):
    df_modified.iloc[row, col] = np.nan

print(f"   - Missing values added: {missing_count}")
print(f"   - Percentage: {(df_modified.isna().sum().sum() / total_cells) * 100:.2f}%")

# 2. ADD DUPLICATES (5% of data)
print("\n2. Adding Duplicates...")
num_duplicates = int(len(df_modified) * 0.05)
duplicate_indices = np.random.choice(df_modified.index, num_duplicates, replace=False)
duplicates = df_modified.loc[duplicate_indices].copy()
df_modified = pd.concat([df_modified, duplicates], ignore_index=True)

print(f"   - Duplicate rows added: {num_duplicates}")
print(f"   - New dataset shape: {df_modified.shape}")

# 3. ADD OUTLIERS
print("\n3. Adding Outliers...")
print(f"   - Columns with outliers: {OUTLIER_COLUMNS}")
outlier_count = 0

# Add outliers to specified columns
for col in OUTLIER_COLUMNS:
    if col in df_modified.columns:
        outlier_indices = np.random.choice(df_modified.index, OUTLIERS_PER_COLUMN, replace=False)
        
        # Add low outliers (first half)
        for idx in outlier_indices[:OUTLIERS_PER_COLUMN//2]:
            if pd.notna(df_modified.loc[idx, col]):
                df_modified.loc[idx, col] = np.random.choice([0, 2, 5, 8, 10])  # Extremely low
                outlier_count += 1
        
        # Add high outliers (second half)
        for idx in outlier_indices[OUTLIERS_PER_COLUMN//2:]:
            if pd.notna(df_modified.loc[idx, col]):
                df_modified.loc[idx, col] = 100  # Maximum score
                outlier_count += 1

print(f"   - Total outlier values added: {outlier_count}")

# Save to new CSV
df_modified.to_csv('data_with_faults.csv', index=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original dataset shape: {df.shape}")
print(f"Modified dataset shape: {df_modified.shape}")
print(f"\nIssues introduced:")
print(f"  - Missing values: ~{(df_modified.isna().sum().sum() / (df_modified.shape[0] * df_modified.shape[1])) * 100:.2f}%")
print(f"  - Duplicate rows: {df_modified.duplicated().sum()}")
print(f"  - Outliers added: {outlier_count} across columns: {OUTLIER_COLUMNS}")
print(f"\nMissing values per column:")
print(df_modified.isna().sum())
print(f"\nNew CSV saved as 'data_with_faults.csv'")
print("="*60)