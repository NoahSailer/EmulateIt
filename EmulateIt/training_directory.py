import os
class training_directory():
    """bookkeeping class"""
    def __init__(self, train_dir):
        if not train_dir.endswith('/'): train_dir = train_dir + '/'
        if not os.path.exists(train_dir): os.mkdir(train_dir)
        self.train_in  = f"{train_dir}training-data_inputs.npy"
        self.train_out = f"{train_dir}training-data_outputs.npy"
        self.fid_in    = f"{train_dir}fid-data_inputs.npy"
        self.fid_out   = f"{train_dir}fid-data_outputs.npy"
        self.weights   = f"{train_dir}weights.json"
