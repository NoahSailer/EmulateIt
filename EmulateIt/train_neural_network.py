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

def train_neural_network(training_directory, 
                         hidden_layer_sizes=(200,200,200), 
                         activation='tanh'):
    """trains the NN"""
    input_filename = f"{training_directory}training-data_inputs.npy"
    output_filename = f"{training_directory}training-data_outputs.npy"
    weights_filename = f"{training_directory}trained_weights.npy"
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,  
                         solver='lbfgs', 
                         max_iter=5000, 
                         learning_rate='adaptive',
                         tol=1e-6,
                         random_state=42)
    model.fit(np.load(input_filename), np.load(output_filename))
    save_model_weights(model, weights_filename)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a neural network from saved weights.")
    parser.add_argument("training_directory", type=str, help="Path to training data.")
    args = parser.parse_args()
    train_neural_network(args.input_filename, args.output_filename, args.weights_filename)
    
if __name__ == "__main__":
    main()
