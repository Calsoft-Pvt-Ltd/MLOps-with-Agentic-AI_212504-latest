import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder



# Load dataset
df = pd.read_csv('data.csv')
# Splitting Features and Target
X = df.drop('Purchase Amount', axis=1)
y = df['Purchase Amount']

# One-Hot Encoding for City
encoder_oh = OneHotEncoder(sparse=False)
X_onehot = pd.DataFrame(encoder_oh.fit_transform(X[['City']]), columns=encoder_oh.get_feature_names_out(['City']))

# Label Encoding for Age Group
encoder_label = LabelEncoder()
X['Age Group Encoded'] = encoder_label.fit_transform(X['Age Group'])

# Target Encoding for Income Level
encoder_target = TargetEncoder()
X['Income Level Encoded'] = encoder_target.fit_transform(X['Income Level'], y)

# Final Data
X_final = pd.concat([X_onehot, X[['Age Group Encoded', 'Income Level Encoded']]], axis=1)
print(X_final)