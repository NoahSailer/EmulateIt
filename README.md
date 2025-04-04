<div style="display: flex; align-items: flex-start; flex-direction: column">
  <div style="flex-shrink: 0;">
    <img src="https://github.com/NoahSailer/EmulateIt/blob/main/figures/emuditto.png" style="width: 200px; margin-right: 16px; align-self: flex-start" />
  </div>
  <div style="display: flex; flex-direction: column; align-self: flex-start; flex: 1; justify-content: flex-start; padding: 0; margin: 0">
    <h3 style="margin: 0;">EmulateIt</h3>
    <p style="margin-top: 4px;">
      A simple package for emulating whatever your heart desires.
    </p>
  </div>
</div>

To install:
```
pip install -v git+https://github.com/NoahSailer/EmulateIt
```

Here's an example for generating training data:
```
cd path/to/EmulateIt/example
interact                         # launch interactive node
conda activate classyenv         # enviornment for classy & mpi4py
srun -n 128 python matter_pk.py  # generates training data
```
The training data is saved as `mpk_inputs.npy` and `mpk_outputs.npy`. To then train a neural network
you can use the `train-nn [inputs.npy] [outputs.npy] [weights.json]` command within an enviornment 
where `scikit-learn` is installed:
```
conda activate sklearn-env
export OMP_NUM_THREADS=128                                          
srun -n 1 -c 128 train-nn mpk_inputs.npy mpk_outputs.npy mpk_weights.json
```
To test the emulator against CLASS, see `matter_pk_emu.py`.



For now I'm using `scikit-learn` for training and evaluations. In the future we may want to have an 
array of options to choose from, especially since these packages become obsolete fairly quickly.
