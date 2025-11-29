import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('synthetic_regression_data.csv')
df = pd.DataFrame(data)

# Prepare X and y
X = df.drop('target', axis=1)
y = df['target']

# Correlation Matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Remove Features with Low Correlation to Target
threshold = 0.7
target_corr = correlation_matrix['target'][abs(correlation_matrix['target']) > threshold].index.tolist()
target_corr.remove('target')  # Remove target itself from the list
print("Features highly correlated with target:", target_corr)

# Remove Features with High Inter-Feature Correlation (Multicollinearity)
print("\nChecking for multicollinearity (feature-to-feature correlation)...")
high_corr_features = set()
feature_corr = correlation_matrix.drop('target', axis=0).drop('target', axis=1)

for i in range(len(feature_corr.columns)):
    for j in range(i):
        if abs(feature_corr.iloc[i, j]) > 0.5:  # Threshold for multicollinearity
            colname = feature_corr.columns[i]
            high_corr_features.add(colname)
            print(f"  {feature_corr.columns[j]} <-> {colname}: {feature_corr.iloc[i, j]:.3f}")

print(f"\nFeatures to remove due to multicollinearity: {high_corr_features}")
features_to_keep = [col for col in X.columns if col not in high_corr_features]
print(f"Features to keep after correlation filtering: {features_to_keep}")

#RFE - Recursive Feature Elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

# Selected Features
selected_features_bool = rfe.support_
selected_features_rfe = X.columns[selected_features_bool].tolist()
print("\nRFE Selected Features:", selected_features_rfe)

#Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Feature Importance
importance = lasso.coef_
selected_features_indices = [i for i, coef in enumerate(importance) if coef != 0]
selected_features_lasso = [X.columns[i] for i in selected_features_indices]
print("\nLasso Selected Features:", selected_features_lasso)
print("\nLasso Feature Coefficients:")
for feature, coef in zip(X.columns, importance):
    if coef != 0:
        print(f"  {feature}: {coef:.4f}")