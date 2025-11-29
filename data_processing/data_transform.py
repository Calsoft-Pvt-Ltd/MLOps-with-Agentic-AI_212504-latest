import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
"""
1. MIN-MAX SCALING:
   ✓ Range: [0, 1]
   ✓ Use when: Features need to be on same scale (0-1)
   ✓ Best for: Neural networks, distance-based algorithms (KNN, K-Means)
   ✗ Sensitive to outliers

2. STANDARD SCALING:
   ✓ Mean: 0, Std: 1
   ✓ Use when: Features have different units/scales
   ✓ Best for: Linear models, SVM, PCA
   ✗ Doesn't bound values to specific range

3. L2 NORMALIZATION:
   ✓ Each sample (row) scaled to unit length
   ✓ Use when: Direction matters more than magnitude
   ✓ Best for: Text classification, clustering
   ✗ Not for features with different scales

4. LOG TRANSFORMATION:
   ✓ Reduces right skewness
   ✓ Use when: Data is right-skewed, exponential relationships
   ✓ Best for: Revenue, population, prices
   ✗ Cannot handle negative values directly

5. SQUARE ROOT TRANSFORMATION:
   ✓ Moderate skewness reduction
   ✓ Use when: Data is moderately skewed
   ✓ Best for: Count data, rates
   ✗ Only for non-negative values
"""
# Load dataset
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

print("="*60)
print("DATA TRANSFORMATION TECHNIQUES")
print("="*60)

print(f"\nOriginal dataset shape: {df.shape}")

# Select numeric columns for transformation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns for transformation: {numeric_cols}")

# Display original statistics
print("\nOriginal Data Statistics:")
print(df[numeric_cols].describe())

# ========================================
# 1. MIN-MAX SCALING (Normalization)
# ========================================
print("\n" + "="*60)
print("1. MIN-MAX SCALING (Range: 0 to 1)")
print("="*60)

df_minmax = df.copy()
minmax_scaler = MinMaxScaler()

# Apply MinMax scaling
df_minmax[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

print("\n✓ Min-Max Scaling completed")
print(f"  Formula: X_scaled = (X - X_min) / (X_max - X_min)")
print(f"\nScaled data range: [{df_minmax[numeric_cols].min().min():.4f}, {df_minmax[numeric_cols].max().max():.4f}]")
print(f"\nMin-Max Scaled Statistics:")
print(df_minmax[numeric_cols].describe())

# Save
df_minmax.to_csv('data_minmax_scaled.csv', index=False)
print(f"\n✓ Saved to 'data_minmax_scaled.csv'")

# ========================================
# 2. STANDARD SCALING (Standardization)
# ========================================
print("\n" + "="*60)
print("2. STANDARD SCALING (Z-Score Normalization)")
print("="*60)

df_standard = df.copy()
standard_scaler = StandardScaler()

# Apply Standard scaling
df_standard[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])

print("\n✓ Standard Scaling completed")
print(f"  Formula: X_scaled = (X - μ) / σ")
print(f"  where μ = mean, σ = standard deviation")
print(f"\nScaled data mean: {df_standard[numeric_cols].mean().mean():.6f} (should be ~0)")
print(f"Scaled data std: {df_standard[numeric_cols].std().mean():.6f} (should be ~1)")
print(f"\nStandard Scaled Statistics:")
print(df_standard[numeric_cols].describe())

# Save
df_standard.to_csv('data_standard_scaled.csv', index=False)
print(f"\n✓ Saved to 'data_standard_scaled.csv'")

# ========================================
# 3. NORMALIZATION (L2 Norm)
# ========================================
print("\n" + "="*60)
print("3. NORMALIZATION (L2 Norm - Unit Vector)")
print("="*60)

df_normalized = df.copy()
normalizer = Normalizer(norm='l2')

# Apply L2 normalization (row-wise)
df_normalized[numeric_cols] = normalizer.fit_transform(df[numeric_cols])

print("\n✓ L2 Normalization completed")
print(f"  Formula: X_normalized = X / ||X||₂")
print(f"  Each row is scaled to unit length (vector magnitude = 1)")
print(f"\nNormalized Statistics:")
print(df_normalized[numeric_cols].describe())

# Verify: Check if row norms are 1
row_norms = np.sqrt((df_normalized[numeric_cols]**2).sum(axis=1))
print(f"\nRow norms (should be ~1): mean={row_norms.mean():.6f}, std={row_norms.std():.6f}")

# Save
df_normalized.to_csv('data_l2_normalized.csv', index=False)
print(f"\n✓ Saved to 'data_l2_normalized.csv'")

# ========================================
# 4. LOG TRANSFORMATION
# ========================================
print("\n" + "="*60)
print("4. LOG TRANSFORMATION")
print("="*60)

df_log = df.copy()

print("\n✓ Applying log transformation to numeric columns")
print(f"  Formula: X_log = log(X + 1)  [log1p to handle zeros]")

# Apply log transformation (using log1p to handle zeros)
for col in numeric_cols:
    # Check if column has non-negative values
    if df_log[col].min() >= 0:
        df_log[f'{col}_log'] = np.log1p(df_log[col])
        print(f"  - Created '{col}_log'")
    else:
        # For negative values, shift then log
        shift_value = abs(df_log[col].min()) + 1
        df_log[f'{col}_log'] = np.log1p(df_log[col] + shift_value)
        print(f"  - Created '{col}_log' (shifted by {shift_value})")

log_cols = [col for col in df_log.columns if col.endswith('_log')]
print(f"\nLog Transformed Statistics:")
print(df_log[log_cols].describe())

# Save
df_log.to_csv('data_log_transformed.csv', index=False)
print(f"\n✓ Saved to 'data_log_transformed.csv'")

# ========================================
# 5. SQUARE ROOT TRANSFORMATION
# ========================================
print("\n" + "="*60)
print("5. SQUARE ROOT TRANSFORMATION")
print("="*60)

df_sqrt = df.copy()

print("\n✓ Applying square root transformation")
print(f"  Formula: X_sqrt = √X")

for col in numeric_cols:
    if df_sqrt[col].min() >= 0:
        df_sqrt[f'{col}_sqrt'] = np.sqrt(df_sqrt[col])
        print(f"  - Created '{col}_sqrt'")

sqrt_cols = [col for col in df_sqrt.columns if col.endswith('_sqrt')]
print(f"\nSquare Root Transformed Statistics:")
print(df_sqrt[sqrt_cols].describe())

# Save
df_sqrt.to_csv('data_sqrt_transformed.csv', index=False)
print(f"\n✓ Saved to 'data_sqrt_transformed.csv'")

# ========================================
# VISUALIZATION
# ========================================
print("\n" + "="*60)
print("6. VISUALIZATION")
print("="*60)

# Select one column for comparison
sample_col = numeric_cols[0] if numeric_cols else None

if sample_col:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Distribution Comparison: {sample_col}', fontsize=16)
    
    # Original
    axes[0, 0].hist(df[sample_col], bins=30, edgecolor='black')
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Frequency')
    
    # MinMax Scaled
    axes[0, 1].hist(df_minmax[sample_col], bins=30, edgecolor='black', color='orange')
    axes[0, 1].set_title('Min-Max Scaled')
    
    # Standard Scaled
    axes[0, 2].hist(df_standard[sample_col], bins=30, edgecolor='black', color='green')
    axes[0, 2].set_title('Standard Scaled')
    
    # Normalized
    axes[1, 0].hist(df_normalized[sample_col], bins=30, edgecolor='black', color='red')
    axes[1, 0].set_title('L2 Normalized')
    axes[1, 0].set_ylabel('Frequency')
    
    # Log Transformed
    if f'{sample_col}_log' in df_log.columns:
        axes[1, 1].hist(df_log[f'{sample_col}_log'], bins=30, edgecolor='black', color='purple')
        axes[1, 1].set_title('Log Transformed')
    
    # Square Root Transformed
    if f'{sample_col}_sqrt' in df_sqrt.columns:
        axes[1, 2].hist(df_sqrt[f'{sample_col}_sqrt'], bins=30, edgecolor='black', color='brown')
        axes[1, 2].set_title('Square Root Transformed')
    
    plt.tight_layout()
    plt.savefig('transformation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'transformation_comparison.png'")
    plt.show()

# ========================================
# COMPARISON SUMMARY
# ========================================
print("\n" + "="*60)
print("TRANSFORMATION SUMMARY")
print("="*60)

print("="*60)
print(f"Files created:")
print("  1. data_minmax_scaled.csv")
print("  2. data_standard_scaled.csv")
print("  3. data_l2_normalized.csv")
print("  4. data_log_transformed.csv")
print("  5. data_sqrt_transformed.csv")
print("  6. transformation_comparison.png")
print("="*60)