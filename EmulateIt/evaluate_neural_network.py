import numpy as np
import json

def load_model_weights(filename):
    """Loads neural network weights from a JSON file."""
    with open(filename, "r") as f:
        weights = json.load(f)
    return weights

def sigmoid(x):
    """Sigmoid (logistic) activation function."""
    return 1 / (1 + np.exp(-x))

class neural_network_emulator:
    def __init__(self,weights_path):
        self.weights = load_model_weights(weights_path)
    def evaluate(self,inputs):
        """Performs a forward pass of the neural network."""
        layer_output = inputs
        for i, (coef, intercept) in enumerate(zip(self.weights["coefs"], self.weights["intercepts"])):
            layer_output = np.dot(layer_output, np.array(coef)) + np.array(intercept)
            if i < len(self.weights["coefs"]) - 1:  # Apply ReLU only to hidden layers
                layer_output = sigmoid(layer_output)
        return layer_output