# ================== DECISION TREE REGRESSOR ==================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

pd.set_option('display.max_columns', None)
print("Libraries imported successfully")

# Load your CSV file
df = pd.read_csv('dataset.csv')

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print(df.head())

print("\nMissing Values:\n", df.isnull().sum())

# Exploratory Data Analysis (EDA)
print(df.info())
print("\nBasic Statistics:\n", df.describe())

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df['area'], df['price'], alpha=0.6)
plt.xlabel('area (sq.ft)')
plt.ylabel('price')
plt.title('area vs price')

plt.subplot(1, 3, 2)
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price by Bedrooms')

plt.subplot(1, 3, 3)
sns.boxplot(x='bathrooms', y='price', data=df)
plt.title('Price by Bathrooms')


# Data Preprocessing
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical columns encoded successfully!", flush=True)
print("Final columns:", df.columns.tolist())

# Features for model
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
           'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
           'parking', 'prefarea', 'furnishingstatus']

X = df[features]
y = df['price']

print("Features selected:", features)
print("Target: price")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# Using same features as before
X = df[features]
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeRegressor(
    max_depth=5,          # Prevents overfitting
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

model.fit(X_train, y_train)
print("Decision Tree Model Trained")

# Predictions
dt_predictions = model.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

print("Decision Tree Performance:")
print(f"Mean Squared Error : {dt_mse:,.2f}")
print(f"R² Score           : {dt_r2:.4f} ({dt_r2*100:.2f}%)")

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully as model.pkl")
