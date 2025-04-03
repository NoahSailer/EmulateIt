from mpi4py import MPI
import numpy as np

def generate_training_data(num_samples, input_bounds, function_to_evaluate):
    """Generates training data within given input bounds."""
    dim = len(input_bounds)
    inputs = np.random.uniform(
        low=[b[0] for b in input_bounds], 
        high=[b[1] for b in input_bounds],
        size=(num_samples, dim))
    outputs = np.array([function_to_evaluate(x) for x in inputs])
    return inputs, outputs

def make_training_data(num_samples, input_bounds, function_to_evaluate, input_filename, output_filename):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0: print(f"Running {size} processes",flush=True)
    samples_per_proc = num_samples // size
    local_inputs, local_outputs = generate_training_data(samples_per_proc, input_bounds, function_to_evaluate)
    gathered_inputs = comm.gather(local_inputs, root=0)
    gathered_outputs = comm.gather(local_outputs, root=0)
    if rank == 0:
        all_inputs = np.vstack(gathered_inputs)
        all_outputs = np.vstack(gathered_outputs)
        np.save(input_filename, all_inputs)
        np.save(output_filename, all_outputs)
        print(f"Saved {len(all_inputs)} training samples to {input_filename} and {output_filename}.",flush=True)