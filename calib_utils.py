# Standard libraries
import numpy as np
from scipy.ndimage import distance_transform_edt
from functools import partial
import nibabel as nib
import json
import ants
from scipy.stats import linregress
import registration as reg

# Constants
GAMMA = 2.6752218708e8  # Gyro Magnetic Ratio - rad/s/T

# Functions

def shrink_mask_mm(mask: np.ndarray, voxel_spacing, margin_mm: float) -> np.ndarray:
    """
    Shrink a mask by keeping only voxels at least margin_mm away from the boundary.
    Uses an Euclidean distance transform with anisotropic sampling.
    """
    mask = mask.astype(bool)
    if margin_mm <= 0:
        return mask

    # distance inside the mask to the nearest background voxel
    dist_in = distance_transform_edt(mask, sampling=voxel_spacing)
    mask_shrunk = np.zeros_like(mask).astype(bool) 
    mask_shrunk[dist_in >= margin_mm] = 1

    return mask_shrunk


def apply_mask(data, mask):

    _mask = mask.astype(bool)

    output_data = np.zeros_like(data)
    output_data[_mask] = data[_mask]

    return output_data


def bmap_3d(s0: np.ndarray, s: np.ndarray, adc: float, mask=None):

    if mask is None:
        mask = np.ones_like(s, bool)
    else:
        mask = mask.astype(bool)

    valid = mask & (s>0) & (s0>0)
    bmap = np.zeros_like(s, dtype=float)
    bmap[valid] = np.log(s0[valid] / s[valid]) / adc

    return bmap
    

def ridge_regression(X, Y, lam=1):

    N, P = X.shape
    
    # Calculate analytical solution to L2-regularised gradient descent
    XtX = X.T @ X
    A_ridge = np.linalg.solve(XtX + lam*np.eye(P), X.T @ Y)

    return A_ridge


def x_from_gvec(gvec):
    """
    Features:
      [g_x, g_y, g_z, g_x^2, g_y^2, g_z^2]
    """
    gvec = np.asarray(gvec, dtype=np.float32)

    x = np.zeros(6, dtype=np.float32)
    x[0:3] = gvec
    x[3:6] = gvec**2

    return x


def z_from_x(x, sigma):

    z = np.zeros_like(x)
    z = x / sigma

    return z


def calc_calib_maps(gvecs, cmaps, mask, as_image=True):

    N = gvecs.shape[0]
    eps = 1e-12

    _mask = mask.astype(bool)

    X = np.apply_along_axis(x_from_gvec, arr=gvecs, axis=1)
    
    # Z-score design matrix - don't apply to first row as defeats the purpose of having a constant term
    sigma = X.std(axis=0, ddof=0, keepdims=True).squeeze()
    sigma = np.where(sigma < eps, 1.0, sigma) # Deal with rows where data is missing e.g. all zeros

    Z = np.apply_along_axis(partial(z_from_x, sigma=sigma), arr=X, axis=1)

    Y = (cmaps[_mask]).T # This broadcasts the 4D gmaps array into a 2D array of shape N x V
    
    A = ridge_regression(Z, Y, lam=0.01)

    return A, sigma


def A_as_nifti(A, mask, affine, sigma, P: int = 9):

    _mask = mask.astype(bool)
        
    A_img = np.zeros((*_mask.shape, P))
    A_img[_mask==1] = A.T

    A_nii = nib.Nifti1Image(A_img, affine=affine)

    calib_metadata = {"sigma": sigma.tolist()}
    
    ext = nib.nifti1.Nifti1Extension(40, json.dumps(calib_metadata).encode("utf-8"))
    A_nii.header.extensions.append(ext)
    return A_nii


def unpack_A_nii(A_path):
    A_nii = nib.load(A_path)
    A_img = A_nii.get_fdata()

    aff = A_nii.affine
    spacing = reg.spacing_from_affine(aff)

    mask = A_img[..., 0] != 0
    A = A_img[mask].T

    calib_metadata = next(json.loads(ext.content.decode("utf-8")) for ext in A_nii.header.extensions if ext.get_code() == 40)
    sigma = calib_metadata["sigma"]

    return A, mask, sigma, spacing


def center_crop_3d(arr, target_shape):
    """
    Crop a 3D array around its center to target_shape.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    if any(t > s for t, s in zip(target_shape, arr.shape)):
        raise ValueError(
            f"target_shape {target_shape} cannot be larger than arr.shape {arr.shape}"
        )

    starts = [(s - t) // 2 for s, t in zip(arr.shape, target_shape)]
    ends = [start + t for start, t in zip(starts, target_shape)]

    return arr[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2],
    ]

def crop_to_overlap(arr1, arr2):
    """
    Center-crop two 3D arrays to their overlapping region, assuming:
    1) same voxel size
    2) same physical center
    """
    if arr1.ndim != 3 or arr2.ndim != 3:
        raise ValueError("Both inputs must be 3D arrays")

    overlap_shape = tuple(min(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))

    arr1_crop = center_crop_3d(arr1, overlap_shape)
    arr2_crop = center_crop_3d(arr2, overlap_shape)

    return arr1_crop, arr2_crop


def bspline_estimation(data, spacing, mask):

    # Convert to Ants
    img_a = ants.from_numpy(data, spacing)
    mask_a = ants.get_mask(ants.from_numpy(data, spacing))
    mask_bool = mask.copy().astype(bool)
    
    # calculate bias field using ants
    bias_field_a = ants.n4_bias_field_correction(img_a, 
                                                 mask=mask_a, 
                                                 rescale_intensities=False,
                                                convergence={'iters':[30, 20, 10, 5], 'tol':1e-6},
                                                shrink_factor=2, 
                                                spline_param=25, 
                                                return_bias_field=True)
    
    # Convert from ants images to numpy arrays
    bias_field = bias_field_a.numpy()
    
    # Use Scipy to then fit the bias field map to the data
    res = linregress(bias_field[mask_bool], data[mask_bool])
    bspline_est = res.slope * bias_field + res.intercept
    
    masked_bspline_est = np.zeros_like(bias_field, dtype=np.float32)
    masked_bspline_est[mask_bool] = bspline_est[mask_bool]

    return masked_bspline_est


def b2grad2(b, delta_small, delta_big, mask=None):

    # b in s/mm^2 -> s/m^2
    b_sm2 = b * 1e6

    denom = GAMMA**2 * delta_small**2 * (delta_big - delta_small / 3)

    if mask is None:
        g = np.sqrt(b_sm2 / denom)   # T/m
    else:
        _mask = mask.astype(bool)
        g = np.zeros_like(b, dtype=float)
        g[_mask] = np.nan_to_num(b_sm2[_mask] / denom, nan=0, posinf=0, neginf=0)

    return np.nan_to_num(g, nan=0, posinf=0, neginf=0)


def grad22b(grad2, delta_small, delta_big, mask=None):
    
    factor = GAMMA**2 * delta_small**2 * (delta_big - delta_small / 3)

    if mask is None:
        b_sm2 = factor * grad2   # s/m^2
    else:
        _mask = mask.astype(bool)
        b_sm2 = np.zeros_like(grad2, dtype=float)
        b_sm2[_mask] = factor * grad2[_mask]
    
    b = np.nan_to_num(b_sm2 / 1e6, nan=0, posinf=0, neginf=0)

    return b


def est_bmap(gvec, delta_small, delta_big, A, sigma, mask):

    mask_bool = mask.astype(bool)
    
    gvec = np.asarray(gvec).reshape(1,-1)
    x = x_from_gvec(gvec)
    z = z_from_x(x, sigma)
    pred_g2maps = np.zeros_like(mask_bool, dtype=float)
    pred_g2maps[mask_bool] = (z @ A).T
    pred_bmaps = grad22b(pred_g2maps, delta_small=delta_small, delta_big=delta_big, mask=mask_bool)

    return pred_bmaps


def minus_bd_map(s, s0, mask=None):

    if not (s.shape == s0.shape == mask.shape):
        raise ValueError(f"Shape mismatch: s {s.shape}, s0 {s0.shape}, mask {mask.shape}")

    if mask is None:
        _mask = reg.get_mask(s0, mask_type=bool)
    else:
        _mask = mask.astype(bool)
    valid = (_mask) & (s0>0) & (s>0)

    minus_bd_map = np.zeros_like(s)
    minus_bd_map[valid] = np.log(s[valid] / s0[valid])

    return minus_bd_map
