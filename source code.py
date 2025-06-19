# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/Nitin5C7/Nithin/refs/heads/main/Housing_Price.csv"  # Replace with your actual URL
data = pd.read_csv(url)

# Rename columns to 'price' and 'area'
data = data.rename(columns={'Hours': 'area', 'Scores': 'price'})

# Display the first few rows of the dataset
print(data.head())

# Create a figure with two subplots (2 rows, 1 column)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

# First subplot (Top): Initial scatter plot
ax1.scatter(data['area'], data['price'], color='blue', label='Data points')
ax1.set_title('Area vs Price')
ax1.set_xlabel('Area')
ax1.set_ylabel('Price')
ax1.legend()

# Splitting the dataset into features (X) and target (y)
X = data[['area']]  # Feature
y = data['price']   # Target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Second subplot (Bottom): Regression line with test data
ax2.scatter(X_test, y_test, color='blue', label='Actual values')
ax2.plot(X_test, y_pred, color='red', label='Regression line')
ax2.set_title('Linear Regression: Test Data')
ax2.set_xlabel('Area')
ax2.set_ylabel('Price')
ax2.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()