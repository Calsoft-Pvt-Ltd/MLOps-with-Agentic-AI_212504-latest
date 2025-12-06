import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder



# Load dataset
df = pd.read_csv('data.csv')
# Splitting Features and Target
X = df.drop('math_score', axis=1)
y = df['math_score']

# One-Hot Encoding for City
encoder_oh = OneHotEncoder(sparse_output=False, drop='first')
X_onehot = pd.DataFrame(encoder_oh.fit_transform(X[['gender']]), columns=encoder_oh.get_feature_names_out(['gender']))
print(X_onehot.head())
# Label Encoding for Age Group
encoder_label = LabelEncoder()
X['parental_level_of_education Encoded'] = encoder_label.fit_transform(X['parental_level_of_education'])
print(X[['parental_level_of_education', 'parental_level_of_education Encoded']].head())
# Target Encoding for Income Level
encoder_target = TargetEncoder()
X['test_preparation_course Encoded'] = encoder_target.fit_transform(X['test_preparation_course'], y)
print(X[['test_preparation_course', 'test_preparation_course Encoded']].head())
# Final Data
X_final = pd.concat([X_onehot, X[['parental_level_of_education']]], axis=1)
print(X_final)
