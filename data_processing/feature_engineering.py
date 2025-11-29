import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv('data.csv')

print("="*60)
print("FEATURE ENGINEERING")
print("="*60)
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Display original data
print("\nOriginal Data (first 5 rows):")
print(df.head())

# ========================================
# 1. DOMAIN-DRIVEN FEATURES
# ========================================
print("\n" + "="*60)
print("1. DOMAIN-DRIVEN FEATURES")
print("="*60)

# Assuming we have score columns: math_score, reading_score, writing_score
# Create meaningful features based on domain knowledge

# Average score across all subjects
df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
print("\n✓ Created 'average_score' - mean of all test scores")

# Total score
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
print("✓ Created 'total_score' - sum of all test scores")

# Score variance/consistency (student performance consistency)
df['score_variance'] = df[['math_score', 'reading_score', 'writing_score']].var(axis=1)
print("✓ Created 'score_variance' - measure of performance consistency")

# Score range (difference between best and worst subject)
df['score_range'] = df[['math_score', 'reading_score', 'writing_score']].max(axis=1) - \
                    df[['math_score', 'reading_score', 'writing_score']].min(axis=1)
print("✓ Created 'score_range' - difference between highest and lowest score")

# Binary features: Did student pass all subjects? (assuming 50 is passing)
df['passed_all'] = ((df['math_score'] >= 50) & 
                    (df['reading_score'] >= 50) & 
                    (df['writing_score'] >= 50)).astype(int)
print("✓ Created 'passed_all' - binary indicator if student passed all subjects")

# Performance level categorization
def categorize_performance(score):
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Average'
    else:
        return 'Poor'

df['performance_category'] = df['average_score'].apply(categorize_performance)
print("✓ Created 'performance_category' - categorical performance level")

# ========================================
# 2. INTERACTION TERMS
# ========================================
print("\n" + "="*60)
print("2. INTERACTION FEATURES")
print("="*60)

# Multiply features to capture interactions
# Math and Reading interaction

df['verbal_to_math_ratio'] = (df['reading_score'] + df['writing_score']) / (2 * df['math_score'] + 1)
print("✓ Created 'verbal_to_math_ratio' - (reading + writing) ÷ (2 × math)")

# ========================================
# 3. POLYNOMIAL FEATURES
# ========================================
print("\n" + "="*60)
print("3. POLYNOMIAL FEATURES")
print("="*60)

# Select numeric columns for polynomial transformation

'''
# Original 3 features (degree 1):
1. math_score
2. reading_score  
3. writing_score

# Squared terms (degree 2):
4. math_score²
5. reading_score²
6. writing_score²

# Interaction/Cross terms (degree 2):
7. math_score × reading_score
8. math_score × writing_score
9. reading_score × writing_score
'''
numeric_cols = ['math_score', 'reading_score', 'writing_score']
X = df[numeric_cols]

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)

# Get feature names
poly_feature_names = poly.get_feature_names_out(numeric_cols)

# Create DataFrame with polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

print(f"\n✓ Created {len(poly_feature_names)} polynomial features (degree=2)")
print(f"  Original features: {numeric_cols}")
print(f"  Polynomial features include:")
print(f"    - {poly_df.columns}")

# Manually add squared terms to main dataframe for clarity
df['math_score_squared'] = df['math_score'] ** 2
df['reading_score_squared'] = df['reading_score'] ** 2
df['writing_score_squared'] = df['writing_score'] ** 2

print("\n✓ Added squared features to main dataframe:")
print("  - math_score_squared")
print("  - reading_score_squared")
print("  - writing_score_squared")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original features: {df.shape[1] - len([c for c in df.columns if c not in list(pd.read_csv('data.csv').columns)])}")
print(f"New features created: {len([c for c in df.columns if c not in list(pd.read_csv('data.csv').columns)])}")
print(f"Final dataset shape: {df.shape}")


print("="*60)