import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Replace 'your_train_data.parquet' with your file path
df = pd.read_csv('HousingData.csv')

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# separate feature variables and target
X_train = train_df.drop(columns=['ID', 'medv'])  # Drop target column 'medv'
X_test = test_df.drop(columns=['ID','medv'])
y_train = train_df['medv']
y_test = test_df['medv']

# Inspect the data
print(df.head())

def standardize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std

X_train_scaled, train_mean, train_std = standardize_features(X_train)
X_test_scaled = (X_test - train_mean) / train_std

X_train_scaled = X_train_scaled.values
X_test_scaled = X_test_scaled.values
y_train = y_train.values
y_test = y_test.values

def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b


def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, w, b, learning_rate, n_iterations):
    m = len(y)
    cost_history = []

    for i in range(n_iterations):
        predictions = X.dot(w) + b
        w_gradient = (1/m) * X.T.dot(predictions - y)
        b_gradient = (1/m) * np.sum(predictions - y)
        
        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient
        
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return w, b, cost_history

n_features = X_train_scaled.shape[1]
w, b = initialize_parameters(n_features)
learning_rate = 0.01
n_iterations = 1000

w, b, cost_history = gradient_descent(X_train_scaled, y_train, w, b, learning_rate, n_iterations)


def predict(X, w, b):
    return X.dot(w) + b

# Make predictions
y_train_pred = predict(X_train_scaled, w, b)
y_test_pred = predict(X_test_scaled, w, b)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train MSE: {mse_train}")
print(f"Test MSE: {mse_test}")
print(f"Train R²: {r2_train}")
print(f"Test R²: {r2_test}")

plt.figure(figsize=(12, 6))
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.show()

# Plotting the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Train Set: Actual vs Predicted Prices')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Test Set: Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')

plt.tight_layout()
plt.show()
