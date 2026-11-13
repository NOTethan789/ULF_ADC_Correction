"""
MP-PCA denoising of 4D diffusion data.
"""

import numpy as np
import mpdenoise as mp


def apply_mppca(data, default_kernel=True, kernel:str=None):

    # Select patch size
    if default_kernel:
        kernel="5,5,5"

    else:
        if kernel is None:
            n_vols = data.shape[-1]
            pr = max(2, int(np.ceil((n_vols ** (1/3) - 1) / 2)))
            k = pr * 2 + 1
            kernel = f"{k},{k},{k}"
            print(f"Applying MPPCA to {n_vols}, hence kernel is [{kernel}]")

    
    # Apply MPPCA
    denoised_data, _, _ = mp.denoise(data, kernel=kernel)
    return denoised_data
