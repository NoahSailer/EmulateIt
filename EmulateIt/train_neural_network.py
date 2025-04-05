from EmulateIt.training_directory import training_directory
from sklearn.neural_network       import MLPRegressor
import numpy as np
import json
import argparse

def save_model_weights(model, filename):
    """Saves the neural network weights as a JSON file."""
    weights = {"coefs": [coef.tolist() for coef in model.coefs_], 
               "intercepts": [intercept.tolist() for intercept in model.intercepts_]}
    with open(filename, "w") as f:
        json.dump(weights, f)
    print(f"Model weights saved to {filename}")

def train_neural_network(train_dir, 
                         hidden_layer_sizes=(200,200,200), 
                         activation='tanh'):
    """trains the NN"""
    tdir  = training_directory(train_dir)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,  
                         solver='lbfgs', 
                         max_iter=5000, 
                         learning_rate='adaptive',
                         tol=1e-6,
                         random_state=42)
    model.fit(np.load(tdir.train_in), np.load(tdir.train_out))
    save_model_weights(model, tdir.weights)

def main():
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument("train_dir", type=str, help="Path to training data.")
    args = parser.parse_args()
    train_neural_network(args.train_dir)
    
if __name__ == "__main__":
    main()
