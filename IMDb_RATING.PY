# =============================================================================
# IMDb Movie Rating Prediction Model
#
# Description:
# This script loads the 'IMDb Movies India.csv' dataset, cleans and
# preprocesses the data, and then trains a RandomForestRegressor model
# to predict movie ratings based on features like year, duration, genre,
# director, actors, and votes. The script also evaluates the model's
# performance and visualizes the results.
#
# Author: Gemini
# Date: 2024-05-24
# =============================================================================

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- 2. Load the Dataset ---
try:
    # Load the dataset from the provided CSV file.
    # The encoding is set to 'latin1' to handle potential special characters.
    df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
    print("Dataset loaded successfully.")
    print("---------------------------------")
except FileNotFoundError:
    print("Error: 'IMDb Movies India.csv' not found.")
    print("Please make sure the dataset file is in the same directory as this script.")
    exit()

# --- 3. Data Cleaning and Preprocessing ---
print("Starting data cleaning and preprocessing...")
print(f"Original dataset shape: {df.shape}")
# print("\nInitial missing values:\n", df.isnull().sum())

# Drop the 'Name' column as it's a unique identifier and not useful for general prediction
df.drop('Name', axis=1, inplace=True)

# Handle missing values
# For categorical columns, fill missing values with 'Unknown'
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].fillna('Unknown')

# Drop rows where 'Rating' or other critical numeric columns have null values
df.dropna(subset=['Rating', 'Duration', 'Year', 'Votes'], inplace=True)

# Clean and convert data types for consistency
# 'Year': Remove parentheses and convert to integer
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(int)

# 'Duration': Remove ' min' suffix and convert to integer
df['Duration'] = df['Duration'].astype(str).str.replace(r'\D', '', regex=True).astype(int)

# 'Votes': Remove commas and convert to integer
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(int)

# 'Genre': For simplicity, we'll use the primary genre listed
df['Genre'] = df['Genre'].apply(lambda x: x.split(',')[0].strip())

print(f"Dataset shape after cleaning: {df.shape}")
print("Data types after conversion:\n", df.dtypes)
print("---------------------------------")


# --- 4. Feature Engineering (Label Encoding) ---
# Convert categorical string features into a numerical format that the model can understand
print("Applying Label Encoding to categorical features...")
le = LabelEncoder()
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# print("\nFirst 5 rows of the fully processed data:\n", df.head())
print("---------------------------------")

# --- 5. Model Building ---
print("Building the prediction model...")
# Define the features (X) and the target variable (y)
X = df.drop('Rating', axis=1)
y = df['Rating']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize and train the RandomForestRegressor model
# n_estimators: number of trees in the forest
# random_state: ensures reproducibility
# n_jobs=-1: use all available CPU cores for faster training
print("\nTraining the RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
model.fit(X_train, y_train)
print("Model training complete.")
print("---------------------------------")

# --- 6. Model Evaluation ---
# Make predictions on the unseen test set
y_pred = model.predict(X_test)

# Calculate standard regression metrics to evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("---------------------------------")
print("(R² of 1.0 is a perfect model, 0.0 means the model performs no better than the mean)")


# --- 7. Feature Importance Analysis ---
# Identify which features had the most impact on the model's predictions
print("Analyzing feature importances...")
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))

# Visualize feature importances for better interpretation
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Feature Importances in Movie Rating Prediction', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
print("\nDisplaying feature importance plot...")
plt.show()

# --- 8. Prediction vs. Actuals Visualization ---
# Create a scatter plot to visualize how well the predicted ratings match the actual ratings
plt.figure(figsize=(10, 8))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5, 'color': 'blue'}, line_kws={'color':'red', 'lw':2})
plt.title('Actual vs. Predicted Movie Ratings', fontsize=16)
plt.xlabel('Actual Ratings', fontsize=12)
plt.ylabel('Predicted Ratings', fontsize=12)
plt.grid(True)
plt.tight_layout()
print("Displaying actual vs. predicted ratings plot...")
plt.show()

print("\nScript finished.")
