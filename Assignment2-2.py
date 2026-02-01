#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np

# Load Dataset using csv module
def load_data(filename):
    X = []
    y = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  

        for row in reader:
            # Skip rows with missing values
            if '' in row or 'NA' in row:
                continue

            row = list(map(float, row))
            X.append(row[:-1])   # Features
            y.append(row[-1])    # Target

    return np.array(X), np.array(y)

# Normalization
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Add Bias Term
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Loss Function
def compute_loss(y, y_pred):
    epsilon = 1e-9  # To avoid log(0)
    return -np.mean(
        y * np.log(y_pred + epsilon) +
        (1 - y) * np.log(1 - y_pred + epsilon)
    )


# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate=0.01, epochs=2000):
    m, n = X.shape
    weights = np.zeros(n)

    for i in range(epochs):
        z = X @ weights
        y_pred = sigmoid(z)

        gradient = (1 / m) * (X.T @ (y_pred - y))
        weights -= learning_rate * gradient

        if i % 200 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {i} | Loss: {loss:.4f}")

    return weights


# Prediction Function
def predict(X, weights, threshold=0.5):
    probabilities = sigmoid(X @ weights)
    return (probabilities >= threshold).astype(int)

# Accuracy Calculation
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Main Function
def main():
    # Load dataset
    X, y = load_data("/Users/stutimahajan/Downloads/framingham.csv")

    print("Dataset shape:", X.shape)

    # Normalize features
    X = normalize(X)

    # Add bias term
    X = add_bias(X)

    # Train Logistic Regression model
    weights = gradient_descent(X, y)

    # Predictions
    y_pred = predict(X, weights)

    # Accuracy
    acc = accuracy(y, y_pred)
    print("\nTraining Accuracy:", acc)

# Run Program
if __name__ == "__main__":
    main()

