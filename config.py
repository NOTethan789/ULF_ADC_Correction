"""
Settings shared by the scripts and notebooks of the project (preprocessing/, analysis/, scanner_characterisation/).

Change a value here and it changes everywhere, so the estimation script and all of the
notebooks stay consistent. Importing this module also puts src/ on sys.path (so the
modules there, e.g. calib_utils, can be imported) and defines absolute paths, so scripts
work whatever directory they are run from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"    # all datasets (not stored on GitHub)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

# Which registered scans the calibration is estimated from and evaluated on:
#   "noisy"    - registered noisy scans. Used so that the training and test data of the
#                leave-one-out analyses are independent (MP-PCA was applied jointly across repeats).
#   "denoised" - registered MP-PCA denoised scans (kept for comparison only)
SCAN_TYPE = "noisy"
SCAN_FILES = {"noisy": "RAW_REG",
              "denoised": "RAW_MPPCA_REG"}

# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------

B0_THRESHOLD = 5     # b-values below this are b=0 scans
SHRINK_DIST_MM = 6   # inset of the calibration mask from the threshold based mask (avoids boundary artefacts)
SPLINE_PARAM = 25    # B-spline smoothing of the gradient maps (spline_param of the N4 bias field fit)
B_EVAL = 700         # b-value from which scans are evaluated in the b=800 figures

# --------------------------------------------------------------------------------------
# Calber Phantom ADC vials
# --------------------------------------------------------------------------------------

CALIBER_ADC_LABELS = {"H2O":[4, 3, 1, 11],
                      "10% PVP": [5, 12],
                      "20% PVP": [6, 13],
                      "30% PVP": [7, 14],
                      "40% PVP": [8, 9],
                      "50% PVP": [2, 10]}

# --------------------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------------------

CALIBRATION_DIR = DATA_DIR / "calibration"
DERIV_DIR = CALIBRATION_DIR / "derivatives"
CALIB_DIR = CALIBRATION_DIR / "calib_temp"   # calibration maps for applying to target data (rename when happy)
PLOT_DIR = CALIBRATION_DIR / "plots_temp"    # figures (rename when happy)

# Target datasets the calibration is applied to
CALIBER_DIR = DATA_DIR / "caliber_phantom"
INVIVO_DIR = DATA_DIR / "invivo"

# Second scanner (b=0 and b=800 scans along each axis, repeated), used to compare the scanners
SCANNER2_DIR = DATA_DIR / "scanner_2"

# Scanner characterisation data (scanner_characterisation/)
SNR_DIR = DATA_DIR / "snr"
B0_MAPPING_DIR = DATA_DIR / "b0_mapping"

# --------------------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------------------

PARAM_FNAME = DERIV_DIR / "calibration_params.h5"
SCANS_FNAME = DERIV_DIR / f"calibration_{SCAN_FILES[SCAN_TYPE]}.nii.gz"
MASK_FNAME = DERIV_DIR / "calibration_MASK.nii.gz"

# Per-scan maps saved by estimate_calibration.py. Volume i of each file corresponds to row i of the
# parameter file (as for the scan files); volumes of b=0 scans are zero.
BMAP_FNAME = DERIV_DIR / f"calibration_BMAP_{SCAN_TYPE}.nii.gz"
GMAP2_FNAME = DERIV_DIR / f"calibration_GMAP2_BSPLINE_{SCAN_TYPE}.nii.gz"


def a_dur_fname(duration):
    """File name of the calibration maps for a gradient duration (s)."""
    return CALIB_DIR / f"calibration_A_DUR_{int(round(duration * 1e3))}.nii.gz"


# Scanner 2 (preprocess_scanner2.ipynb). Same scan type as the calibration scans.
SCANNER2_DERIV_DIR = SCANNER2_DIR / "derivatives"
SCANNER2_PARAM_FNAME = SCANNER2_DERIV_DIR / "scanner_2_params.h5"
SCANNER2_SCANS_FNAME = SCANNER2_DERIV_DIR / f"scanner_2_{SCAN_FILES[SCAN_TYPE]}.nii.gz"
SCANNER2_MASK_FNAME = SCANNER2_DERIV_DIR / "scanner_2_MASK.nii.gz"


A_FULL_FNAME = CALIB_DIR / "calibration_A_FULL.nii.gz"
CALIB_INFO_FNAME = CALIB_DIR / "calibration_calib_info.json"
