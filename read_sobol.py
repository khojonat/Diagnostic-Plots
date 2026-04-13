import numpy as np

def read_sobol(run):

    basePath = f'/project/torrey-group/jkho/FIRE_Mass_varied/hyperparam_files/sobol_params.txt'

    sobol = np.loadtxt(basePath,skiprows=1)

    return sobol[run]