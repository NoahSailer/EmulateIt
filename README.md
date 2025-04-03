# EmuateIt

To install:
```
pip install -v git+https://github.com/NoahSailer/EmulateIt
```

Here's an example for generating training data and training a neural network on Perlmutter:
```
cd path_to_EmulateIt/example
interact                               # launch interactive node
conda activate classyenv               # activate whatever conda enviornment has CLASS installed
srun -n 32 -c 4 python matter_pk.py    # generates training data
conda activate nntrainer               # activate conda enviornment for training
srun -n 32 -c 4 train-nn mpk_inputs.npy mpk_outputs.npy mpk_weights.json # train the network and save weights
python matter_pk_emu.py                # example evaluation of neural network to get matter power spectrum
```