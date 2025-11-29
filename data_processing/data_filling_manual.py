import pandas as pd

# Sample dataset
data = pd.read_csv('data_with_faults.csv')
df = pd.DataFrame(data)
print(type(df))

# original data
print("Original Data:")
print(df)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df['math_score'] = df['math_score'].fillna(df['math_score'].mean())  # Replace with mean math_score
df['writing_score'] = df['writing_score'].fillna(df['writing_score'].median())  # Replace with median writing_score
df['parental_level_of_education'] = df['parental_level_of_education'].fillna('some college')  # Replace None with 'No Activity'

# cleaned data
print("\nCleaned Data:")
print(df)

