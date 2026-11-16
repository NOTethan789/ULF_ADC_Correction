"""
Estimate the calibration maps from the calibration scans and save them for applying to target data.

For every diffusion-weighted scan:
    1. the b-value map is calculated from the signal loss relative to the preceding b=0 scan,
       assuming the known diffusivity of water at the scan temperature
    2. the b-value map is converted to a squared-gradient map (Stejskal-Tanner equation)
    3. the squared-gradient map is smoothed with a B-spline (N4 bias field fit)
Calibration maps (6 coefficient maps per voxel: g_x, g_y, g_z, g_x^2, g_y^2, g_z^2) are then fitted by ridge
regression of the smoothed gradient maps against the gradient vectors of the scans:
    - at the duration level: one set per gradient duration (fitted to the scans of that duration)
    - at the scanner level: one set fitted to all scans

Outputs
    derivatives/<testname>_BMAP_<scan type>.nii.gz          per-scan b-value maps
    derivatives/<testname>_GMAP2_BSPLINE_<scan type>.nii.gz per-scan smoothed squared-gradient maps
        Volume i of each file corresponds to row i of the parameter file (as for the scan files); volumes of
        b=0 scans are zero. The b-value, duration, direction and sequence of every volume, and the settings
        used, are also stored in the NIfTI header (extension) as JSON.
    results/calibration_maps/<testname>_A_DUR_<ms>.nii.gz, <testname>_A_FULL.nii.gz   calibration maps
    results/calibration_maps/<testname>_calib_info.json     settings and number of scans used for each fit

The leave-one-out analyses do not use the saved calibration maps (they refit without the held-out repeat,
see calib_utils.loo_bmap_predictions) but do use the per-scan maps saved here.

Settings are in config.py. Run from anywhere:  python analysis/estimate_calibration.py
"""

import json
from functools import partial

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root, for config.py
import config
import file_io as fio
import registration as reg
import temperature as temp
import calib_utils as cal


def save_per_scan_map(array_4d, fname, affine, meta):
    """Save a [x, y, z, scan] array as NIfTI with the metadata JSON in the header."""
    nii = nib.Nifti1Image(array_4d.astype(np.float32), affine=affine)
    nii.header.extensions.append(nib.nifti1.Nifti1Extension(40, json.dumps(meta).encode("utf-8")))
    nib.save(nii, fname)
    print(f"{fname.name} saved")


# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
config.RESULTS_CALIB_DIR.mkdir(parents=True, exist_ok=True)

print(f"Estimating calibration from the {config.SCAN_TYPE} scans: {config.SCANS_FNAME.name}")

params_df = pd.read_hdf(config.PARAM_FNAME, key="df", mode="r")

scans_nii = nib.load(config.SCANS_FNAME)
scan_aff = scans_nii.affine
scan_spacing = reg.spacing_from_affine(scan_aff)

if scans_nii.shape[-1] != params_df.shape[0]:
    raise ValueError(f"{scans_nii.shape[-1]} scans but {params_df.shape[0]} rows in the parameter file")

df = params_df.copy()
df["row"] = np.arange(df.shape[0])    # position in the parameter file and scan files
df["image"] = list(np.transpose(scans_nii.get_fdata(), [3,0,1,2]))
df["adc"] = df["temperature"].apply(partial(temp.water_diffusivity, polynomial=True))

preprocess_mask = nib.load(config.MASK_FNAME).get_fdata().astype(bool)

# Create a mask slightly inset from threshold based mask to avoid boundary artefacts.
core_mask = cal.shrink_mask_mm(preprocess_mask, voxel_spacing=scan_spacing, margin_mm=config.SHRINK_DIST_MM)

# ------------------------------------------------------------------------------
# Per-scan maps
# ------------------------------------------------------------------------------
df.sort_values("scan_time", inplace=True, ignore_index=True)

# Put the b=0 image on b=0 rows, NaN elsewhere
df["s0_image"] = df["image"].where(df["b_value"] < config.B0_THRESHOLD)

# Forward-fill so each DWI scan gets the most recent preceding b=0 image
df["s0_image"] = df["s0_image"].ffill()

# Keep only diffusion-weighted scans that have a preceding b=0 scan
dwi_df = df[df["b_value"] > config.B0_THRESHOLD].copy()
dwi_df = dwi_df[dwi_df["s0_image"].notna()].copy()
print(f"{dwi_df.shape[0]} diffusion-weighted scans ({df.shape[0] - dwi_df.shape[0]} b=0 scans or scans without a preceding b=0)")

dwi_df["bmap"] = dwi_df.apply(lambda row: cal.bmap_3d(s0=row["s0_image"],
                                                      s=row["image"],
                                                      adc=row["adc"],
                                                      mask=preprocess_mask),
                                                      axis=1)

# Estimate gradient maps from b-value maps assuming Stejskal-Tanner Equation applies.
dwi_df["gmap2"] = dwi_df.apply(lambda row: cal.b2grad2(b=row["bmap"],
                                                       delta_small=row["dwi_grad_duration"],
                                                       delta_big=(row["TE"] / 2),
                                                       mask=core_mask),
                                                       axis=1)

# Estimate a smoothed version of each gradient map for fitting
tqdm.pandas(desc="B-spline smoothing of the gradient maps")
dwi_df["gmap2_bspline"] = dwi_df.progress_apply(lambda row: cal.bspline_estimation(data=row["gmap2"],
                                                                                   spacing=row["spacing"],
                                                                                   mask=core_mask,
                                                                                   spline_param=config.SPLINE_PARAM),
                                                                                   axis=1)

# ------------------------------------------------------------------------------
# Save the per-scan maps, in the order of the parameter file
# ------------------------------------------------------------------------------
n_rows = params_df.shape[0]
bmap_4d = np.zeros((*core_mask.shape, n_rows), dtype=np.float32)
gmap2_4d = np.zeros((*core_mask.shape, n_rows), dtype=np.float32)
for _, row in dwi_df.iterrows():
    bmap_4d[..., row["row"]] = row["bmap"]
    gmap2_4d[..., row["row"]] = row["gmap2_bspline"]

is_dwi = np.zeros(n_rows, dtype=bool)
is_dwi[dwi_df["row"].values] = True

meta = {"description": "Volume i corresponds to row i of the parameter file; volumes of b=0 scans are zero",
        "scan_type": config.SCAN_TYPE,
        "scans_file": config.SCANS_FNAME.name,
        "b0_threshold": config.B0_THRESHOLD,
        "shrink_dist_mm": config.SHRINK_DIST_MM,
        "spline_param": config.SPLINE_PARAM,
        "volume_labels": [{"row": i,
                           "seq_name": str(params_df["seq_name"].iloc[i]),
                           "b_value": float(params_df["b_value"].iloc[i]),
                           "dwi_grad_duration": float(params_df["dwi_grad_duration"].iloc[i]),
                           "dwi_direction": str(params_df["dwi_direction"].iloc[i]),
                           "is_dwi": bool(is_dwi[i])} for i in range(n_rows)]}

save_per_scan_map(bmap_4d, config.BMAP_FNAME, scan_aff, meta)
save_per_scan_map(gmap2_4d, config.GMAP2_FNAME, scan_aff, meta)

# ------------------------------------------------------------------------------
# Fit the calibration maps
# ------------------------------------------------------------------------------
def fit_calibration(subset_df, fname):

    input_gvec = np.stack(subset_df["grad_vec"].values, axis=0).astype(np.float32)
    input_gmaps = np.stack(subset_df["gmap2_bspline"].values, axis=-1).astype(np.float32)

    A, sigma = cal.calc_calib_maps(input_gvec, input_gmaps, core_mask, as_image=False)
    A_nii = cal.A_as_nifti(A, mask=core_mask, affine=scan_aff, sigma=sigma, P=6)
    fio.save_nifti(A_nii, fname.parent, fname.name)


n_scans_used = {}

# Duration level
for dur in sorted(dwi_df["dwi_grad_duration"].unique()):
    subset_df = dwi_df[dwi_df["dwi_grad_duration"] == dur]
    fit_calibration(subset_df, config.a_dur_fname(dur))
    n_scans_used[config.a_dur_fname(dur).name] = int(subset_df.shape[0])

# Scanner level
fit_calibration(dwi_df, config.A_FULL_FNAME)
n_scans_used[config.A_FULL_FNAME.name] = int(dwi_df.shape[0])

info = {k: v for k, v in meta.items() if k != "volume_labels"}
info["n_scans_used"] = n_scans_used
with open(config.CALIB_INFO_FNAME, "w") as f:
    json.dump(info, f, indent=2)
print(f"{config.CALIB_INFO_FNAME.name} saved")
