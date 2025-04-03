# EmuateIt

To install:
```
pip install -v git+https://github.com/NoahSailer/EmulateIt
```

Here's an example for generating training data:
```
cd path_to_EmulateIt/example
interact                               # launch interactive node
conda activate classyenv               # activate whatever conda enviornment has CLASS/mpi4py installed
srun -n 128 python matter_pk.py        # generates training data
python matter_pk_emu.py                # example evaluation of neural network to get matter power spectrum
```
The training data is saved as `mpk_inputs.npy` and `mpk_outputs.npy`. To then train a neural network
you can use the `train-nn` command:
```
conda activate nntrainer                                             
srun -n 128 train-nn mpk_inputs.npy mpk_outputs.npy mpk_weights.json # train the network and save weights
```

For now I'm using `scikit-learn` for training/evaluations. In the future we may want to have an 
array of options to choose from, especially since these packages become obsolete fairly quickly.