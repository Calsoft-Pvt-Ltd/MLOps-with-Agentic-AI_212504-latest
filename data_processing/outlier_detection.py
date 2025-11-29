import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Sample dataset
data = pd.read_csv('data_with_faults.csv')
df = pd.DataFrame(data)

print("="*60)
print("OUTLIER DETECTION METHODS")
print("="*60)

# Check missing values first
print(f"\nTotal rows: {len(df)}")
print(f"Missing values in math_score: {df['math_score'].isna().sum()}")
print(f"Non-missing values in math_score: {df['math_score'].notna().sum()}")

# Visualize
print("\n1. Visualizing Outliers with Boxplot...")
sns.boxplot(df['math_score'])
plt.title("Math Scores with Outliers")
plt.show()

# METHOD 1: IQR (Interquartile Range)
print("\n2. IQR Method")
print("-" * 40)
Q1, Q3 = np.percentile(df['math_score'].dropna(), [25, 75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

# Separate into three groups: valid data, outliers, and missing
valid_data_iqr = df[(df['math_score'] >= lower_bound) & (df['math_score'] <= upper_bound)]
outliers_iqr = df[(df['math_score'] < lower_bound) | (df['math_score'] > upper_bound)]
missing_data = df[df['math_score'].isna()]

print(f"\nOriginal data: {len(df)} rows")
print(f"Valid data (IQR): {len(valid_data_iqr)} rows")
print(f"Outliers detected (IQR): {len(outliers_iqr)} rows")
print(f"Missing values: {len(missing_data)} rows")
print(f"Sum check: {len(valid_data_iqr)} + {len(outliers_iqr)} + {len(missing_data)} = {len(valid_data_iqr) + len(outliers_iqr) + len(missing_data)}")

# METHOD 2: Z-Score
print("\n3. Z-Score Method")
print("-" * 40)
threshold = 3  # Typically use 3 for outlier detection

# Create a full z-score column (with NaN for missing values)
df['z_score'] = np.nan
df.loc[df['math_score'].notna(), 'z_score'] = stats.zscore(df['math_score'].dropna())

print(f"Z-Score threshold: {threshold}")
print(f"Mean: {df['math_score'].mean():.2f}")
print(f"Std Dev: {df['math_score'].std():.2f}")

# Separate into three groups
valid_data_zscore = df[np.abs(df['z_score']) <= threshold]
outliers_zscore = df[np.abs(df['z_score']) > threshold]

print(f"\nOriginal data: {len(df)} rows")
print(f"Valid data (Z-Score): {len(valid_data_zscore)} rows")
print(f"Outliers detected (Z-Score): {len(outliers_zscore)} rows")
print(f"Missing values: {len(missing_data)} rows")
print(f"Sum check: {len(valid_data_zscore)} + {len(outliers_zscore)} + {len(missing_data)} = {len(valid_data_zscore) + len(outliers_zscore) + len(missing_data)}")

# Display outliers
print("\nOutliers detected by Z-Score:")
print(outliers_zscore[['math_score', 'z_score']].dropna())

# Drop the z_score column from final dataframes
valid_data_zscore = valid_data_zscore.drop('z_score', axis=1)
df = df.drop('z_score', axis=1)

# Compare both methods
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"IQR Method - Outliers removed: {len(outliers_iqr)}")
print(f"Z-Score Method - Outliers removed: {len(outliers_zscore)}")
print(f"\nBreakdown:")
print(f"  Valid + Outliers + Missing = Total")
print(f"  {len(valid_data_iqr)} + {len(outliers_iqr)} + {len(missing_data)} = {len(df)}")