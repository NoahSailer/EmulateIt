import numpy as np
import json
from sklearn.neural_network import MLPRegressor

def save_model_weights(model, filename):
    """Saves the neural network weights as a JSON file."""
    weights = {"coefs": [coef.tolist() for coef in model.coefs_], 
               "intercepts": [intercept.tolist() for intercept in model.intercepts_]}
    with open(filename, "w") as f:
        json.dump(weights, f)
    print(f"Model weights saved to {filename}")

def train_neural_network(input_filename, output_filename, model_filename):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(np.load(input_filename), np.load(output_filename))
    save_model_weights(model, model_filename)