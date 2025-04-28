import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/ASUS/Downloads/survey.csv')  # Change to your dataset path

# Check the first few rows to ensure it's loaded properly
print(data.head())
missing_data = data.isnull().sum()
print(f"Missing values per column:\n{missing_data}")

# Impute missing values with the most frequent value for categorical columns
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Impute missing values with the mean for numerical columns
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    data[col] = data[col].fillna(data[col].mean())

# Step 2: Encode Categorical Variables
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Step 3: Scale Numerical Features
# Identify numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Standardize numerical features using StandardScaler
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 4: Prepare the dataset for training
# Assuming 'seek_help' is the target column
X = data.drop(columns=['seek_help', 'Timestamp'])  # Features
y = data['seek_help']  # Target variable

# Step 5: Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)  # Initialize Logistic Regression model
log_reg.fit(X_train, y_train)  # Train the model

# Step 7: Make predictions
y_pred = log_reg.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Show detailed classification metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Evaluate model coefficients for insight into feature importance
print("Model Coefficients:", log_reg.coef_)

# Step 9: Plot the Accuracy Graph (Training vs Test)
train_accuracy = log_reg.score(X_train, y_train)  # Training accuracy
test_accuracy = accuracy  # Test accuracy

# Accuracy Bar Graph
plt.figure(figsize=(6, 4))
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['green', 'blue'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Step 10: Plot Confusion Matrix
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Step 1: Handle Missing Values (same as before)
# ... (handling missing data and other preprocessing steps)

# Step 2: Encode Categorical Variables (same as before)
# Step 3: Scale Numerical Features (same as before)
# Step 4: Prepare the dataset for training

# Split the data and train the Logistic Regression model as shown before

# Step 5: Plot the Accuracy Graph and Confusion Matrix (shown earlier)

# Ensure this line is executed in your environment
plt.show()
  # This will ensure that the plot appears if running in a script
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors (you can tune this parameter)
knn.fit(X_train, y_train)  # Train the model

# Step 7: Make predictions
y_pred_knn = knn.predict(X_test)

# Step 8: Evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Model Accuracy: {accuracy_knn*100:.2f}%")

# Show detailed classification metrics for KNN
print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 9: Train Logistic Regression for comparison
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Step 10: Make predictions with Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# Step 11: Evaluate Logistic Regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg*100:.2f}%")

# Show detailed classification metrics for Logistic Regression
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Confusion Matrix for Logistic Regression
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 12: Compare KNN with Logistic Regression
print("\nModel Comparison:")
print(f"KNN Accuracy: {accuracy_knn*100:.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_log_reg*100:.2f}%")
