from EmulateIt.training_directory import training_directory 
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

def make_training_data(num_samples, input_fid, input_bounds, function_to_evaluate, train_dir='./', norm_type=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    tdir = training_directory(train_dir)
    output_fid = function_to_evaluate(input_fid)
    if norm_type == 'division': new_function = lambda x: function_to_evaluate(x)/output_fid
    else:                       new_function = lambda x: function_to_evaluate(x)
    if rank == 0: 
        print(f"Running {size} processes",flush=True)
        np.save(tdir.fid_in,input_fid)
        np.save(tdir.fid_out,output_fid)
    samples_per_proc = num_samples // size
    local_inputs, local_outputs = generate_training_data(samples_per_proc, input_bounds, new_function)
    gathered_inputs = comm.gather(local_inputs, root=0)
    gathered_outputs = comm.gather(local_outputs, root=0)
    if rank == 0:
        all_inputs = np.vstack(gathered_inputs)
        all_outputs = np.vstack(gathered_outputs)
        np.save(tdir.train_in, all_inputs)
        np.save(tdir.train_out, all_outputs)
        print(f"Saved {len(all_inputs)} training samples to {tdir.train_in} and {tdir.train_out}.",flush=True)
