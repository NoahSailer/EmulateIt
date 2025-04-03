from EmulateIt.make_training_data import make_training_data
import numpy as np
from classy import Class

kval = np.logspace(-3,1,5000) # [h/Mpc]
z    = 0.5                    # redshift

def get_matter_pk(cosmo_params):
    """returns the matter power spectrum given cosmo_params=log1e10As,ns,omc,omb,h"""
    log1e10As,ns,omc,omb,h = cosmo_params
    params = {'output': 'mPk','P_k_max_h/Mpc': 10.,'z_pk': '0.0,10',
              'A_s': np.exp(log1e10As)*1e-10,'n_s': ns,'h': h, 'N_ur': 1.0196,
              'N_ncdm': 2,'m_ncdm': '0.01,0.05','tau_reio': 0.0568,
              'omega_b': omb,'omega_cdm': omc}
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return np.array([cosmo.pk_cb(k*h,z)*h**3. for k in kval])

num_samples     = int(1e4)
input_bounds    = [(2,4),(0.95,1.05),(0.05,0.15),(0.005,0.02),(0.5,0.9)]
input_filename  = 'mpk_inputs.npy'
output_filename = 'mpk_outputs.npy'
make_training_data(num_samples, input_bounds, get_matter_pk, input_filename, output_filename)