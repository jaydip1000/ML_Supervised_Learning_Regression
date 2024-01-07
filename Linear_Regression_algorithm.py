import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def linear_regression(x_train, y_train, learning_rate=0.01, epochs=1000, plot=False):
    # Add a bias term to features
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    
    # Initialize weights with zeros
    weights = np.zeros(x_train.shape[1])

    # Gradient Descent
    for epoch in range(epochs):
        # Calculate predictions
        y_pred = np.dot(x_train, weights)

        # Calculate gradient
        gradient = np.dot(x_train.T, (y_pred - y_train)) / len(y_train)

        # Update weights
        weights -= learning_rate * gradient

        # Visualize the progress if plot=True
        if plot and epoch % 100 == 0:
            plt.scatter(x_train[:, 1], y_train, label='Actual')
            plt.plot(x_train[:, 1], y_pred, color='red', label='Predicted')
            plt.title(f'Linear Regression - Epoch {epoch}')
            plt.xlabel('Feature')
            plt.ylabel('Output')
            plt.legend()
            plt.show()

    return weights

# Example usage:
# Assuming you have a dataset with one feature and the corresponding output
# x_train, y_train are NumPy arrays or Pandas Series

''' # Example:
# x_train = np.array([1, 2, 3, 4, 5])
# y_train = np.array([2, 4, 5, 4, 5])

# Train the linear regression model
weights = linear_regression(x_train, y_train, learning_rate=0.01, epochs=1000, plot=True)

# Make predictions on new data
x_test = np.array([6, 7, 8])
x_test_with_bias = np.column_stack((np.ones(len(x_test)), x_test))
y_pred = np.dot(x_test_with_bias, weights)

print("Weights:", weights)
print("Predictions:", y_pred) '''

