import numpy as np
import json
import argparse
from sklearn.neural_network import MLPRegressor

def save_model_weights(model, filename):
    """Saves the neural network weights as a JSON file."""
    weights = {"coefs": [coef.tolist() for coef in model.coefs_], 
               "intercepts": [intercept.tolist() for intercept in model.intercepts_]}
    with open(filename, "w") as f:
        json.dump(weights, f)
    print(f"Model weights saved to {filename}")

def train_neural_network(input_filename, output_filename, weights_filename, hidden_layer_sizes=(100,), max_iter=500):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(np.load(input_filename), np.load(output_filename))
    save_model_weights(model, weights_filename)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a neural network from saved weights.")
    parser.add_argument("input_filename", type=str, help="Path to the .npy file containing input data.")
    parser.add_argument("output_filename", type=str, help="Path to save the evaluation output as a .npy file.")
    parser.add_argument("weights_filename", type=str, help="Path to the JSON file containing model weights.")
    args = parser.parse_args()
    train_neural_network(args.input_filename, args.output_filename, args.weights_filename)
    
if __name__ == "__main__":
    main()
