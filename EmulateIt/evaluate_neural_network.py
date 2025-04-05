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
    def __init__(self,training_directory,norm_type=None):
        """
        training_directory = str ending with /
        norm_type = None, 'division', ...
        """
        self.weights = load_model_weights(f"{training_directory}trained_weights.json")
        self.fid_input = np.load(f"{training_directory}fid_input.npy")
        self.fid_output = np.load(f"{training_directory}fid_output.npy")
        self.norm_type = norm_type
    def evaluate(self,inputs):
        """Performs a forward pass of the neural network."""
        layer_output = inputs
        if self.norm_type == 'division': layer_output = inputs/self.fid_input
        for i, (coef, intercept) in enumerate(zip(self.weights["coefs"], self.weights["intercepts"])):
            layer_output = np.dot(layer_output, np.array(coef)) + np.array(intercept)
            if i < len(self.weights["coefs"]) - 1:  # Apply ReLU only to hidden layers
                layer_output = np.tanh(layer_output)
        if self.norm_type == 'division': return layer_output*self.fid_output
        return layer_output
