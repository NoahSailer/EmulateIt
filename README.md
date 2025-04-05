### EmulateIt

<table style="border: 5px solid #2b3137; border-radius: 8px;">
  <tr>
    <td style="border: none; vertical-align: top;">
      <img src="https://raw.githubusercontent.com/NoahSailer/EmulateIt/main/figures/emuditto.png" alt="Emuditto" width="130"/>
    </td>
    <td style="border: none; vertical-align: top;">
      Emulate whatever your ❤️ desires. Install with:<br>
      <pre>pip install -v git+https://github.com/NoahSailer/EmulateIt</pre>
    </td>
  </tr>
</table>

Here's an example for generating training data:
```
cd path/to/EmulateIt/example
interact                         # launch interactive node
conda activate classyenv         # enviornment for classy & mpi4py
srun -n 128 python matter_pk.py  # generates (normalized) training data
```
The training data is saved as `example/training-data_[inputs][outputs].npy` in the 
specified directory (in this case: `example/`). You can use the `train-nn [train_dir/]` 
command within an enviornment where `scikit-learn` is installed to train the NN:
```
conda activate sklearn-env
export OMP_NUM_THREADS=128                                          
srun -n 1 -c 128 train-nn ./
```
The weights are saved `example/weights.json`. To test the emulator against `CLASS` see 
`example/matter_pk_emu.py`.


For now I'm using `scikit-learn` for training and evaluations. In the future we may want to have an 
array of options to choose from, especially since these packages become obsolete fairly quickly.
