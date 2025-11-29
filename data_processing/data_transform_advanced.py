import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from scipy import stats
import matplotlib.pyplot as plt

# Generate exponential data
np.random.seed(42)
original_data = np.random.exponential(size=1000)

print("="*60)
print("EXPONENTIAL DATA TRANSFORMATION")
print("="*60)

# Original data statistics
print("\nOriginal Data Statistics:")
print(f"Mean: {original_data.mean():.4f}")
print(f"Std: {original_data.std():.4f}")
print(f"Min: {original_data.min():.4f}")
print(f"Max: {original_data.max():.4f}")
print(f"Skewness: {pd.Series(original_data).skew():.4f}")

# ========================================
# 1. MIN-MAX SCALING
# ========================================
print("\n" + "="*60)
print("1. MIN-MAX SCALING")
print("="*60)

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(original_data.reshape(-1, 1)).flatten()

print(f"\nMin-Max Scaled Statistics:")
print(f"Mean: {minmax_scaled.mean():.4f}")
print(f"Std: {minmax_scaled.std():.4f}")
print(f"Min: {minmax_scaled.min():.4f}")
print(f"Max: {minmax_scaled.max():.4f}")
print(f"Range: [{minmax_scaled.min():.4f}, {minmax_scaled.max():.4f}]")

# ========================================
# 2. STANDARD SCALING
# ========================================
print("\n" + "="*60)
print("2. STANDARD SCALING")
print("="*60)

standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(original_data.reshape(-1, 1)).flatten()

print(f"\nStandard Scaled Statistics:")
print(f"Mean: {standard_scaled.mean():.6f} (should be ~0)")
print(f"Std: {standard_scaled.std():.6f} (should be ~1)")
print(f"Min: {standard_scaled.min():.4f}")
print(f"Max: {standard_scaled.max():.4f}")
print(f"Skewness: {pd.Series(standard_scaled).skew():.4f}")

# ========================================
# 3. LOG TRANSFORMATION
# ========================================
print("\n" + "="*60)
print("3. LOG TRANSFORMATION")
print("="*60)

# Using log1p (log(1 + x)) to handle zeros
log_transformed = np.log1p(original_data)

print(f"\nLog Transformed Statistics:")
print(f"Mean: {log_transformed.mean():.4f}")
print(f"Std: {log_transformed.std():.4f}")
print(f"Min: {log_transformed.min():.4f}")
print(f"Max: {log_transformed.max():.4f}")
print(f"Skewness: {pd.Series(log_transformed).skew():.4f} (reduced from {pd.Series(original_data).skew():.4f})")

# ========================================
# 4. BOX-COX TRANSFORMATION
# ========================================
print("\n" + "="*60)
print("4. BOX-COX TRANSFORMATION")
print("="*60)

# Box-Cox requires positive values (all our exponential data is positive)
boxcox_transformed, lambda_param = stats.boxcox(original_data)

print(f"\nBox-Cox Transformed Statistics:")
print(f"Optimal lambda: {lambda_param:.4f}")
print(f"  (λ=0 → log transform, λ=1 → no transform, λ=0.5 → sqrt)")
print(f"Mean: {boxcox_transformed.mean():.4f}")
print(f"Std: {boxcox_transformed.std():.4f}")
print(f"Min: {boxcox_transformed.min():.4f}")
print(f"Max: {boxcox_transformed.max():.4f}")
print(f"Skewness: {pd.Series(boxcox_transformed).skew():.4f}")

# ========================================
# 5. YEO-JOHNSON TRANSFORMATION (handles negative values)
# ========================================
print("\n" + "="*60)
print("5. YEO-JOHNSON TRANSFORMATION")
print("="*60)

# Yeo-Johnson is similar to Box-Cox but works with negative values
pt = PowerTransformer(method='yeo-johnson', standardize=True)
yeojohnson_transformed = pt.fit_transform(original_data.reshape(-1, 1)).flatten()

print(f"\nYeo-Johnson Transformed Statistics:")
print(f"Optimal lambda: {pt.lambdas_[0]:.4f}")
print(f"Mean: {yeojohnson_transformed.mean():.6f}")
print(f"Std: {yeojohnson_transformed.std():.6f}")
print(f"Min: {yeojohnson_transformed.min():.4f}")
print(f"Max: {yeojohnson_transformed.max():.4f}")
print(f"Skewness: {pd.Series(yeojohnson_transformed).skew():.4f}")

# ========================================
# 6. LOG + STANDARD SCALING (Combined)
# ========================================
print("\n" + "="*60)
print("6. LOG + STANDARD SCALING (Combined)")
print("="*60)

# First log transform, then standard scale
log_then_standard = standard_scaler.fit_transform(log_transformed.reshape(-1, 1)).flatten()

print(f"\nLog + Standard Scaled Statistics:")
print(f"Mean: {log_then_standard.mean():.6f}")
print(f"Std: {log_then_standard.std():.6f}")
print(f"Min: {log_then_standard.min():.4f}")
print(f"Max: {log_then_standard.max():.4f}")
print(f"Skewness: {pd.Series(log_then_standard).skew():.4f}")

# ========================================
# VISUALIZATION
# ========================================
print("\n" + "="*60)
print("7. VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Exponential Data Transformations', fontsize=16)

# Original
axes[0, 0].hist(original_data, bins=50, edgecolor='black', color='skyblue')
axes[0, 0].set_title(f'Original\n(Skew: {pd.Series(original_data).skew():.2f})')
axes[0, 0].set_ylabel('Frequency')

# Min-Max Scaled
axes[0, 1].hist(minmax_scaled, bins=50, edgecolor='black', color='orange')
axes[0, 1].set_title(f'Min-Max Scaled\n(Range: [0, 1])')

# Standard Scaled
axes[0, 2].hist(standard_scaled, bins=50, edgecolor='black', color='green')
axes[0, 2].set_title(f'Standard Scaled\n(Mean: 0, Std: 1)')

# Log Transformed
axes[1, 0].hist(log_transformed, bins=50, edgecolor='black', color='red')
axes[1, 0].set_title(f'Log Transformed\n(Skew: {pd.Series(log_transformed).skew():.2f})')
axes[1, 0].set_ylabel('Frequency')

# Box-Cox
axes[1, 1].hist(boxcox_transformed, bins=50, edgecolor='black', color='purple')
axes[1, 1].set_title(f'Box-Cox (λ={lambda_param:.2f})\n(Skew: {pd.Series(boxcox_transformed).skew():.2f})')

# Yeo-Johnson
axes[1, 2].hist(yeojohnson_transformed, bins=50, edgecolor='black', color='brown')
axes[1, 2].set_title(f'Yeo-Johnson (λ={pt.lambdas_[0]:.2f})\n(Skew: {pd.Series(yeojohnson_transformed).skew():.2f})')

# Log + Standard
axes[2, 0].hist(log_then_standard, bins=50, edgecolor='black', color='pink')
axes[2, 0].set_title(f'Log + Standard\n(Skew: {pd.Series(log_then_standard).skew():.2f})')
axes[2, 0].set_ylabel('Frequency')

# Box plots comparison
box_data = [original_data, minmax_scaled, standard_scaled, log_transformed, 
            boxcox_transformed, yeojohnson_transformed, log_then_standard]
axes[2, 1].boxplot(box_data, labels=['Orig', 'MinMax', 'Std', 'Log', 'BoxCox', 'YeoJ', 'Log+Std'])
axes[2, 1].set_title('Box Plot Comparison')
axes[2, 1].set_ylabel('Values')
axes[2, 1].tick_params(axis='x', rotation=45)

# Q-Q plots for normality check
from scipy.stats import probplot

# Q-Q plot for Box-Cox
probplot(boxcox_transformed, dist="norm", plot=axes[2, 2])
axes[2, 2].set_title('Q-Q Plot: Box-Cox\n(closer to line = more normal)')

plt.tight_layout()
plt.savefig('exponential_transformations_boxcox.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'exponential_transformations_boxcox.png'")
plt.show()

# ========================================
# SAVE TO CSV
# ========================================
print("\n" + "="*60)
print("8. SAVING RESULTS")
print("="*60)

# Create DataFrame with all transformations
df_results = pd.DataFrame({
    'original': original_data,
    'minmax_scaled': minmax_scaled,
    'standard_scaled': standard_scaled,
    'log_transformed': log_transformed,
    'boxcox_transformed': boxcox_transformed,
    'yeojohnson_transformed': yeojohnson_transformed,
    'log_then_standard': log_then_standard
})

df_results.to_csv('exponential_data_transformed.csv', index=False)
print("✓ All transformations saved to 'exponential_data_transformed.csv'")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*60)
print("SUMMARY - SKEWNESS COMPARISON")
print("="*60)

summary_data = {
    'Transformation': ['Original', 'Min-Max', 'Standard', 'Log', 'Box-Cox', 'Yeo-Johnson', 'Log+Standard'],
    'Skewness': [
        pd.Series(original_data).skew(),
        pd.Series(minmax_scaled).skew(),
        pd.Series(standard_scaled).skew(),
        pd.Series(log_transformed).skew(),
        pd.Series(boxcox_transformed).skew(),
        pd.Series(yeojohnson_transformed).skew(),
        pd.Series(log_then_standard).skew()
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df['Skewness'] = summary_df['Skewness'].round(4)
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print(f"""
1. Original exponential data: Highly right-skewed ({pd.Series(original_data).skew():.2f})

2. Min-Max & Standard Scaling: Preserve skewness, just rescale

3. Log Transformation: Reduces skewness to {pd.Series(log_transformed).skew():.2f}

4. Box-Cox Transformation:
   ✓ Optimal λ = {lambda_param:.4f}
   ✓ Automatically finds best power transformation
   ✓ Skewness reduced to {pd.Series(boxcox_transformed).skew():.2f}
   ✗ Requires positive values only

5. Yeo-Johnson Transformation:
   ✓ Optimal λ = {pt.lambdas_[0]:.4f}
   ✓ Similar to Box-Cox but handles negative values
   ✓ Standardized output (mean=0, std=1)
   ✓ Skewness: {pd.Series(yeojohnson_transformed).skew():.2f}

Recommendation for Exponential Data:
✓ Use BOX-COX or YEO-JOHNSON for optimal skewness reduction
✓ Both automatically find the best transformation parameter
✓ Yeo-Johnson is more flexible (handles negative values)
""")

print("="*60)