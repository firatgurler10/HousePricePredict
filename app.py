import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing

#1. Loading Data
california = fetch_california_housing(data_home='C:/Users/Fırat/Desktop/sklearn_data')
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['PRICE'] = california.target

#2. Discover the data
print(california_df.head())
print(california_df.describe())

#3. Data Visualization (Correlation Map)
plt.figure(figsize=(15,10))
heatmap = sns.heatmap(california_df.corr(), annot=True, cmap='coolwarm')

# Rotation of text on X and Y axes
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, horizontalalignment='right')
plt.show()

#4. Feature and Target Variable Distinction
X = california_df.drop(columns=['PRICE']) 
y = california_df['PRICE']

# 5. Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Training the Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Making Predictions
y_pred = model.predict(X_test)

# 8. Evaluating Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) score: {r2}")

# 9. Visualizing Prediction Results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Estimated Prices")
plt.title("Relationship Between Actual and Estimated Prices")
plt.show()