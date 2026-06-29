# Machine Learning Business Cases
# Sales Prediction and Fraud Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1 - SALES PREDICTION (REGRESSION)
# ==========================================================

print("\n===== SALES PREDICTION =====")


# Creating dataset
np.random.seed(42)

months = np.arange(1, 25)

events = np.random.choice(
    [0, 1],
    size=24
)

sales = (
    months * 10
    + events * 40
    + np.random.normal(0, 20, 24)
)


df_sales = pd.DataFrame({
    "Month": months,
    "Sales": sales,
    "Event": events
})


print("\nDataset preview:")
print(df_sales.head())


print("\nDataset information:")
print(df_sales.info())


print("\nStatistical summary:")
print(df_sales.describe())


# Separating variables

X_sales = df_sales[["Month", "Event"]]
y_sales = df_sales["Sales"]


X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales,
    y_sales,
    test_size=0.3,
    random_state=42
)


# Training model

sales_model = LinearRegression()

sales_model.fit(
    X_train_sales,
    y_train_sales
)


# Predictions

sales_predictions = sales_model.predict(
    X_test_sales
)


# Evaluation

mse_sales = mean_squared_error(
    y_test_sales,
    sales_predictions
)

r2_sales = r2_score(
    y_test_sales,
    sales_predictions
)


print("\nSales Model Results:")
print(f"MSE: {mse_sales}")
print(f"R² Score: {r2_sales}")


# Visualization

order = X_test_sales["Month"].argsort()

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test_sales["Month"].iloc[order],
    y_test_sales.iloc[order],
    label="Real Sales"
)

plt.plot(
    X_test_sales["Month"].iloc[order],
    sales_predictions[order],
    label="Predicted Sales"
)

plt.title(
    "Sales Prediction - Linear Regression"
)

plt.xlabel("Month")
plt.ylabel("Sales")

plt.legend()
plt.show()



# ==========================================================
# 2 - FRAUD DETECTION (CLASSIFICATION)
# ==========================================================

print("\n===== FRAUD DETECTION =====")


# Creating dataset

np.random.seed(42)


values = np.random.normal(
    1000,
    200,
    1000
)


locations = np.random.choice(
    ["Location_A", "Location_B", "Location_C"],
    size=1000
)


frauds = np.random.choice(
    [0, 1],
    size=1000
)


df_fraud = pd.DataFrame({
    "Value": values,
    "Location": locations,
    "Fraud": frauds
})


print("\nFraud dataset preview:")
print(df_fraud.head())


# Transform categorical data

df_fraud = pd.get_dummies(
    df_fraud,
    columns=["Location"]
)


# Separating variables

X_fraud = df_fraud.drop(
    "Fraud",
    axis=1
)

y_fraud = df_fraud["Fraud"]


X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud,
    y_fraud,
    test_size=0.3,
    random_state=42
)



# Training model

fraud_model = LogisticRegression(
    max_iter=1000
)


fraud_model.fit(
    X_train_fraud,
    y_train_fraud
)



# Predictions

fraud_predictions = fraud_model.predict(
    X_test_fraud
)



# Evaluation

accuracy = accuracy_score(
    y_test_fraud,
    fraud_predictions
)

matrix = confusion_matrix(
    y_test_fraud,
    fraud_predictions
)

report = classification_report(
    y_test_fraud,
    fraud_predictions
)


print("\nFraud Detection Results:")
print(f"Accuracy: {accuracy}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(report)


print("\nProject completed successfully!")
