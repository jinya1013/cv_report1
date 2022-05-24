import numpy as np

def psnr(true, pred):
    mse = np.mean((true - pred)**2)
    return 10 * np.log10(np.max(true)/mse)