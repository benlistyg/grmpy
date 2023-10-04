# grmpy
Python code for implementing the Graded Response Model (repo name pronounced "grumpy")

# Background

This is a simple implementation of a common Item Response Theory (IRT) model in Python

# Code

``import numpy as np
from scipy.optimize import minimize

class GradedResponseModel:
    def __init__(self, num_categories):
        self.num_categories = num_categories
        self.a = np.random.rand(num_categories - 1)
        self.b = np.random.randn(num_categories)
    
    def probability(self, theta):
        logits = self.a * (theta - self.b[:-1])
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.concatenate([[1.0 - np.sum(probs)], probs])
    
    def log_likelihood(self, theta, response):
        probs = self.probability(theta)
        log_probs = np.log(probs)
        log_likelihood = response * log_probs
        return -np.sum(log_likelihood)
    
    def fit(self, response, initial_theta=0.0):
        result = minimize(
            lambda theta: self.log_likelihood(theta, response),
            initial_theta,
            method='L-BFGS-B'
        )
        self.theta = result.x
    
    def predict(self, theta):
        probs = self.probability(theta)
        return probs

# Example usage
num_categories = 4
response = np.array([1, 0, 0, 0])  # Example item response pattern

model = GradedResponseModel(num_categories)
model.fit(response)  # Fitting to the item response

new_theta = np.random.randn()
predictions = model.predict(new_theta)
print("Predicted probabilities:", predictions)`

```
