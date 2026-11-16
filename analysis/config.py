"""
Settings shared by every script and notebook in analysis/.

Change a value here and it changes everywhere, so the estimation script and all of the
plotting notebooks stay consistent. Importing this module also puts the project root on
sys.path (so the modules in the root, e.g. calib_utils, can be imported) and defines
absolute paths, so scripts work whatever directory they are run from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
# Directories
# --------------------------------------------------------------------------------------

CALIBRATION_DIR = ROOT / "calibration"
DERIV_DIR = CALIBRATION_DIR / "derivatives"
CALIB_DIR = CALIBRATION_DIR / "calib_temp"   # calibration maps for applying to target data (rename when happy)
PLOT_DIR = CALIBRATION_DIR / "plots_temp"    # figures (rename when happy)

# Target datasets the calibration is applied to
CALIBER_DIR = ROOT / "caliber_phantom"
INVIVO_DIR = ROOT / "invivo"

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


A_FULL_FNAME = CALIB_DIR / "calibration_A_FULL.nii.gz"
CALIB_INFO_FNAME = CALIB_DIR / "calibration_calib_info.json"
