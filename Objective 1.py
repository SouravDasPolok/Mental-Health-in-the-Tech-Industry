# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Update path if necessary)
# Correcting the file path
data = pd.read_csv('C:/Users/ASUS/Downloads/survey.csv')
# Standardize Gender column to only have 'Male' and 'Female'
# Replace any variations like 'M' or 'F' with 'Male' and 'Female'

# Clean the Gender column (converting common variations to 'Male' and 'Female')
data['Gender'] = data['Gender'].replace({'M': 'Male', 'F': 'Female', 'male': 'Male', 'female': 'Female'})

# Remove any rows where Gender is not 'Male' or 'Female' (if any)
data = data[data['Gender'].isin(['Male', 'Female'])]

# Check the cleaned data
print(data['Gender'].value_counts())  # Verify the classes in Genderplt.figure(figsize=(8, 6))
sns.countplot(x='treatment', data=data)  # Use the correct column name
plt.title('Treatment for Mental Health')
plt.xlabel('Treatment')
plt.ylabel('Count')
plt.show()


# 1. Gender Distribution (Bar Chart)
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 2. Treatment for Mental Health (Count Plot)
# Use the correct column name based on the output


print(data.columns)
# Step 1: Remove any rows where the Age is negative or extremely high
data = data[data['Age'] > 0]

# Step 2: Ensure 'Seek_help' has valid categories and handle any unexpected values
# Replace any common variations to standardize the 'seek_help' column
data['seek_help'] = data['seek_help'].replace({
    'yes': 'Yes', 'no': 'No', "don't know": "Don't know", 'Yes ': 'Yes', 'No ': 'No'
})

# Step 3: Check for missing values and handle them (e.g., fill or drop)
data = data.dropna(subset=['Age', 'seek_help'])  # Drop rows where Age or Seek_help is missing

# Step 4: Check the cleaned data
print(data[['Age', 'seek_help']].head())
# Remove rows where Age is greater than 100
data = data[data['Age'] <= 100]

# Verify if the rows were removed by checking the unique values in the Age column
print(data['Age'].unique())

# 3. Box Plot to check the relationship between Age and whether people seek help
plt.figure(figsize=(8, 6))
sns.boxplot(x='seek_help', y='Age', data=data)  # Ensure correct column name
plt.title('Age Distribution based on Mental Health Help Seeking')
plt.xlabel('Seek Help')
plt.ylabel('Age')
plt.show()

# 4. Age Distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 5. Correlation Heatmap for Numerical Features
# First, let's select only the numerical columns
# Convert 'Gender' and 'Seek_help' columns to numerical using Label Encoding
# Convert categorical columns into numeric using Label Encoding
# Boxplot to visualize the relationship between Age and treatment
plt.figure(figsize=(8, 6))
sns.boxplot(x='treatment', y='Age', data=data)
plt.title('Age Distribution based on Treatment')
plt.xlabel('Treatment')
plt.ylabel('Age')
plt.show()

# Countplot for the distribution of 'treatment' column
plt.figure(figsize=(8, 6))
sns.countplot(x='treatment', data=data)
plt.title('Distribution of Treatment for Mental Health')
plt.xlabel('Treatment')
plt.ylabel('Count')
plt.show()


