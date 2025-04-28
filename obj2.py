import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'C:/Users/ASUS/Downloads/survey.csv'
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
# Check for missing values
missing_data = data.isnull().sum()
print(f"Missing values per column:\n{missing_data}")

# Option 1: Drop rows with missing values (if appropriate)
data_cleaned = data.dropna()  # Drop rows with any missing values

# Option 2: Impute missing values with the most frequent value (mode) for categorical columns
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# For numerical columns, you can use the mean or median to fill missing values
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    data[col] = data[col].fillna(data[col].mean())

# Step 2: Encode Categorical Variables
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Alternatively, if you want to use one-hot encoding (for non-ordinal categories):
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Step 3: Scale Numerical Features
# Identify numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Standardize numerical features (e.g., Age) using StandardScaler
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 4: Verify Preprocessing
print(f"Preprocessed Data (first 5 rows):\n{data.head()}")

# Step 5: (Optional) Split Data for Model Building
# Assuming 'target' is your column to predict, modify as necessary
# Replace 'target' with your actual target variable column name
# X = data.drop(columns=['target'])  # Features
# y = data['target']  # Target

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
