from EmulateIt.evaluate_neural_network import neural_network_emulator
from matter_pk import get_matter_pk
import numpy as np

cosmo_params = [3.04,0.96]
pk_true = get_matter_pk(cosmo_params)
mpk_emu = neural_network_emulator('mpk_weights.json')
pk_emu  = mpk_emu.evaluate(cosmo_params)

print("Ratio of emulated to true power spectrum:",pk_emu/pk_true)