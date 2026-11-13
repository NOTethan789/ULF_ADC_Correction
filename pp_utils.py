#   --------------------------
#   Libraries
#   --------------------------

# Standard libraries
import numpy as np
import os
import json
import pandas as pd
from glob import glob

from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Specific libraries
import nibabel as nib
import pydicom as pyd
import ants
import tempfile
import shutil
import SimpleITK as sitk

# downloaded files
import mpdenoise as mp

#   --------------------------
# Constants
#   --------------------------

GAMMA = 2.675221900e8 # gyromagnetic ratio of the proton in radians per second per Tesla

#   --------------------------
#   Functions
#   --------------------------

def scrape_files(dir_path: str, filetype = "*"):
    """
    Recursively scrape a directory for files matching one or more filetypes.

    Args:
        dir_path: Directory to search.
        filetype: Filetype(s) to match (e.g. "txt", ["txt", "csv"], "dcm")

    Returns:
        Sorted list of matching file paths
    """

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Normalize filetype to a list
    if isinstance(filetype, str):
        filetypes = [filetype]
    else:
        filetypes = list(filetype)

    all_files = []

    for ft in filetypes:
        # Remove leading dot if present
        if ft.startswith("."):
            ft = ft[1:]

        search_pattern = os.path.join(dir_path, f"**/*.{ft}")
        files = glob(search_pattern, recursive=True)
        all_files.extend(files)

    if not all_files:
        raise ValueError(f"No files found in '{dir_path}' with filetype '{filetype}'")

    return sorted(all_files)


def dicom_to_affine(dicom_path: str):
    
    dcm = pyd.dcmread(dicom_path)

    # Validate multi-frame
    nframes = int(getattr(dcm, "NumberOfFrames", 0))
    if nframes <= 1:
        raise ValueError("This function expects a multi-frame (Enhanced) MR DICOM with NumberOfFrames > 1.")

    if not hasattr(dcm, "PerFrameFunctionalGroupsSequence") or len(dcm.PerFrameFunctionalGroupsSequence) == 0:
        raise ValueError("Missing PerFrameFunctionalGroupsSequence; function expects a multi-frame (Enhanced) DICOM")

    def _get_shared_item(seq_name: str):
        """Return the requested attribute, or None."""
        if not hasattr(dcm, "SharedFunctionalGroupsSequence") or len(dcm.SharedFunctionalGroupsSequence) == 0:
            return None
        shared0 = dcm.SharedFunctionalGroupsSequence[0]
        if not hasattr(shared0, seq_name) or len(getattr(shared0, seq_name)) == 0:
            return None
        return getattr(shared0, seq_name)[0]

    def _get_perframe_item(frame_idx: int, seq_name: str):
        """Return the requested attribute from the requested frame, or None."""
        pf = dcm.PerFrameFunctionalGroupsSequence[frame_idx]
        if not hasattr(pf, seq_name) or len(getattr(pf, seq_name)) == 0:
            return None
        return getattr(pf, seq_name)[0]

    # ImagePositionPatient (origin) from first frame as position is a frame specific parameter
    plane_pos = _get_perframe_item(0, "PlanePositionSequence")
    if plane_pos is None or not hasattr(plane_pos, "ImagePositionPatient"):
        raise ValueError("Could not find ImagePositionPatient at PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].")
    ipp = np.asarray(plane_pos.ImagePositionPatient, dtype=float)

    # ImageOrientationPatient (direction cosines) which is a shared parameter across frames
    plane_orient = _get_shared_item("PlaneOrientationSequence")
    if plane_orient is None or not hasattr(plane_orient, "ImageOrientationPatient"):
        raise ValueError("Could not find ImageOrientationPatient in PlaneOrientationSequence (shared or per-frame).")
    iop = np.asarray(plane_orient.ImageOrientationPatient, dtype=float)  # [r0 r1 r2 c0 c1 c2]

    # Extract PixelSpacing
    if hasattr(dcm, "PixelSpacing"):
        ps = np.asarray(dcm.PixelSpacing, dtype=float)
    else:
        pixel_measures = _get_shared_item("PixelMeasuresSequence")
        if pixel_measures is not None and hasattr(pixel_measures, "PixelSpacing"):
            ps = np.asarray(pixel_measures.PixelSpacing, dtype=float)  # [row_spacing, col_spacing]
        else:
            raise ValueError("Could not find PixelSpacing (Top level parameter or in Shared Functional Parameter)")
        
    # Extract slice thickness
    if hasattr(dcm, "SliceThickness"):
        slice_thickness = float(dcm.SliceThickness)
    else:
        pixel_measures = _get_shared_item("PixelMeasuresSequence")
        if pixel_measures is not None and hasattr(pixel_measures, "SliceThickness"):
            slice_thickness = float(dcm.SliceThickness)
        else:
            raise ValueError("Could not find SliceThickness (Top level parameter or in Shared Functional Parameter)")
    

    # Separate row and column parameters
    row_orient = iop[0:3]
    col_orient = iop[3:6]
    row_spacing = float(ps[0])
    col_spacing = float(ps[1])

    # Calculate slice direction
    slice_orient = np.cross(row_orient, col_orient)
    nrm = float(np.linalg.norm(slice_orient))
    slice_orient /= nrm # Normalise vector

    # Build affine for MRI Scan
    col_axis = row_orient * col_spacing
    row_axis = col_orient * row_spacing
    slice_axis = slice_orient * slice_thickness

    affine = np.eye(4, dtype=float)
    affine[0:3, 0] = slice_axis
    affine[0:3, 1] = row_axis
    affine[0:3, 2] = col_axis
    affine[0:3, 3] = ipp

    return affine


def spacing_from_affine(affine:np.ndarray):
    R = affine[0:3, 0:3] # Spacing & orientation subarray
    spacing = np.linalg.norm(R, axis=0)
    return spacing.tolist()


def save_nifti(nifti, dir_path: str, filename: str):
    save_path = os.path.join(dir_path, filename)
    nib.save(nifti, save_path)
    print(f"{filename} saved")


def save_array(arr, dir_path, fname):
    
    save_path = os.path.join(dir_path, fname)

    test = arr[0]
    if isinstance(test, str):
        fmt = "%s"
    elif isinstance(test, int):
        fmt = "%i"
    elif isinstance(test, float):
        fmt = "%f"
    else:
        raise TypeError("Data in array must be a str, int or float before saving.")

    np.savetxt(save_path, arr, fmt=fmt)
    print(f"{fname} saved")


def ants_mask(data, cleanup=2, fmt=int):
    data_a = ants.from_numpy(data)
    mask_a = ants.get_mask(data_a, cleanup=cleanup)
    return mask_a.numpy().astype(fmt)


def resample_array(img: np.ndarray, current_spacing: np.ndarray, target_spacing: list, interp_type: int = 2):
    """
    Use Antspy to resample an input array to a new voxel spacing

    :param img: Image to be resampled (2D, 3D)
    :param aff: Affine correspondign to the supplied image
    :param target_spacing: 'Target spacing' to resample the image onto.
    :param interp_type: 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
    """
    
    if isinstance(current_spacing, list):
        cs = current_spacing.copy()
    elif isinstance(current_spacing, (tuple, np.ndarray)):
        cs = list(current_spacing)
    else:
        raise TypeError(f"Expect 'current_spacing' of type: list, np.ndarray, tuple. Received type {type(current_spacing)}")
    
    if isinstance(target_spacing, list):
        ts = target_spacing.copy()
    elif isinstance(target_spacing, (tuple, np.ndarray)):
        ts = list(target_spacing)
    else:
        raise TypeError(f"Expect 'target_spacing' of type: list, np.ndarray, tuple. Received type {type(target_spacing)}")
    
    if len(current_spacing) != len(target_spacing) or len(current_spacing) != 3:
        raise ValueError(f"'current_spacing and 'target_spacing' must be both of length 3. Received lengths {len(current_spacing)}, {len(target_spacing)}")
    
    # Resample Image
    img_a = ants.from_numpy(img, spacing=cs)
    img_rs_a = ants.resample_image(img_a, resample_params=target_spacing, use_voxels=False, interp_type=interp_type)
    img_rs = img_rs_a.numpy()

    return img_rs


def dicom_contrast(dicom_fname: str):
    dcm = pyd.dcmread(dicom_fname)
    contrast = dcm.AcquisitionContrast
    return contrast


def dicom_update_scaling_params(dicom_filename):

    dcm = pyd.dcmread(dicom_filename)

    # Extract scaling parameters
    true_min = float(dcm[0x0351, 0x1000].value) # Minimum true intensity
    true_max = float(dcm[0x0351, 0x1001].value) # Maximum true intensity
    uint_x_max = float(dcm[0x0351, 0x1002].value) # Max intensity of stored data
    m = (true_max - true_min) / uint_x_max
    c = true_min

    dcm.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleIntercept = c
    dcm.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleSlope = m
    dcm.save_as(dicom_filename)


def dcm_scaled_image(dicom_fname:str):
    """
    Extract the pixel array from a Hyperfine DICOM file and apply 
    the scaling parameters stored in the DICOM header to convert 
    pixel values to true intensity values.
    Hyperfine DICOMs store the pixel arrays as unsigned intergers 
    ranging from 0 to a maximum value of 4095 (12-bit images).

    :param dicom_fname: DICOM filename
    """

    dcm = pyd.dcmread(dicom_fname)

    # Extract scaling parameters from dicom header
    true_min = float(dcm[0x0351, 0x1000].value) # Minimum true intensity
    true_max = float(dcm[0x0351, 0x1001].value) # Maximum true intensity
    uint_x_max = float(dcm[0x0351, 0x1002].value) # Max intensity of stored data
    m = (true_max - true_min) / uint_x_max
    c = true_min

    data = dcm.pixel_array.astype(np.float32)
    scaled_data = m * data + c

    return scaled_data


def dcm2niix_wrapper(dicom_filename: str) -> nib.Nifti1Image:
    """
    Convert a (single-file) DICOM to a NIfTI image using dcm2niix.

    - Creates a temporary output directory
    - Runs dcm2niix
    - Finds the produced .nii/.nii.gz file and loads it via nibabel
    - Cleans up temp outputs

    :param dicom_filename: Path to input DICOM (or a directory of DICOMs; dcm2niix supports both)
    :return: nibabel Nifti1Image
    """

    if not Path(dicom_filename).exists():
        raise FileNotFoundError(f"DICOM input not found: {dicom_filename}")

    # Optional: only call this if it exists in your environment
    if "dicom_update_scaling_params" in globals():
        dicom_update_scaling_params(str(dicom_filename))

    temp_dir = Path(tempfile.mkdtemp(prefix="dcm2niix_"))

    try:
        cmd = ["dcm2niix",
               "-z", "y",
               "-s", "y",
               "-f", "%i_%d_%s",
               "-o", str(temp_dir),
               str(dicom_filename)]

        subprocess.run(cmd, check=True, text=True, capture_output=True)

        # dcm2niix may write multiple outputs; pick NIfTI files only
        nii_files = sorted(temp_dir.rglob("*.nii.gz"))

        if not nii_files:
            raise RuntimeError(f"dcm2niix produced no NIfTI files in {temp_dir}.")

        # If multiple outputs exist, choose the largest file (often the main image)
        nii_file = max(nii_files, key=lambda p: p.stat().st_size)
        nii = nib.load(str(nii_file))
        nii_stored = nib.Nifti1Image(nii.get_fdata(), nii.affine, nii.header)

        return nii_stored

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def bet2_wrapper(nifti, frac: float = 0.5, robust: bool = True, mask: bool = False) -> nib.Nifti1Image:
    """
    Skull-strip a NIfTI using FSL BET (commonly referred to as "bet2").

    Works with either:
      - a path to a .nii/.nii.gz file, or
      - an 'in-memory' nibabel Nifti1Image

    Strategy:
      - If 'in memory', write nifti file to a temporary directory
      - Runs FSL `bet`
      - Loads the output NIfTI via nibabel and forces voxel data into memory
      - Cleans up temp outputs
      - Returns a Nifti1Image

    Parameters
    ----------
    nifti : str | Path | nib.Nifti1Image
        Input image or path.
    frac : float
        BET fractional intensity threshold (-f). Typical range ~0.1–0.8.
    robust : bool
        Use robust brain centre estimation (-R).
    mask : bool
        Also generate a binary brain mask (-m).

    Returns
    -------
    nib.Nifti1Image
        Skull-stripped image with voxel data loaded into memory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="bet_"))

    try:
        # Prepare input on disk
        if isinstance(nifti, (str, Path)):
            input_path = Path(nifti)
            if not input_path.exists():
                raise FileNotFoundError(f"NIfTI input not found: {input_path}")
        elif isinstance(nifti, nib.Nifti1Image):
            input_path = temp_dir / "input.nii.gz"
            nib.save(nifti, str(input_path))
        else:
            raise TypeError("nifti must be a path (str/Path) or a nib.Nifti1Image")

        out_base = temp_dir / "brain"  # FSL writes brain.nii.gz when given a basename

        cmd = ["bet", str(input_path), str(out_base), "-f", str(frac)]
        if robust:
            cmd.append("-R")
        if mask:
            cmd.append("-m")

        # Run BET
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        except FileNotFoundError as e:
            raise RuntimeError("FSL BET command not found. Make sure FSL is installed and `bet` is on your PATH ") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"BET failed with exit code {e.returncode}.\n") from e

        # BET output is typically brain.nii.gz (or brain.nii)
        candidates = [Path(str(out_base) + ".nii.gz"), Path(str(out_base) + ".nii")]
        out_path = next((p for p in candidates if p.exists()), None)

        if out_path is None:
           raise RuntimeError(f"BET produced no output NIfTI in {temp_dir}.")

        out_img = nib.load(str(out_path))

        # Force voxel data into memory so we can safely delete temp_dir
        data = out_img.get_fdata()
        out_img_stored = nib.Nifti1Image(data, out_img.affine, out_img.header)

        return out_img_stored

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def stack_nifti(nifti_list: list):
    """
    Stack a list of 3D nifti images into a single 4D nifti along the 4th dimension.
    :param nifti_list: List of 3D nifti images.
    """

    for nifti in nifti_list:
        if nifti.affine != nifti_list[0].affine:
            raise ValueError("All nifti images must have the same affine to be combined")
        if nifti.getfdata().ndim != 3:
            raise ValueError("All nifti images must be 3D to be combines")
        if nifti.get_fdata().shape != nifti_list[0].shape:
            raise ValueError("All nifti images must have the same shape to be combined")

    data_4D = np.stack([nifti.get_fdata().astype(np.float32) for nifti in nifti_list], axis=3)
    nifti_4D = nib.Nifti1Image(data_4D, nifti_list[0].affine)

    return nifti_4D


def dicom_patient_id(dicom_fname: str, prefix=None, suffix=None):
    """
    Extract pateint ID from DICOM header, with option to remove
    a common prefix or suffix from the ID string. Prefixes and suffixes are
    commonly used to label a file for a particular project directory.
    
    :param dicom_fname: DICOM filename
    :param prefix: Optional string to remove from start of patient ID
    :param suffix: Optional string to remove from end of patient ID
    """
    
    dcm = pyd.dcmread(dicom_fname)
    id = dcm.PatientID

    if prefix is not None:
        id = id.replace(prefix, "")
    if suffix is not None:
        id = id.replace(suffix, "")

    return id


def dicom_bvalue(dicom_fname: str):

    """
    Extract b-value from DICOM header of a Hyperfine diffusions scan.
    : param dicom_fname: Dicom filename
    """

    dcm = pyd.dcmread(dicom_fname)

    if not dcm.AcquisitionContrast == "DIFFUSION":
        raise ValueError(f"Scan is not a diffusion scan, but has contrast type: {dcm.AcquisitionContrast}")
    
    try:
        bvalue = dcm.SharedFunctionalGroupsSequence[0].MRDiffusionSequence[0].DiffusionBValue
    except:
        print("bvalue parameter not found in DICOM header, labelling as 'UNDEFINED'")
        bvalue = "UNDEFINED"

    return bvalue


def dicom_sequence_params(dicom_fname: str):
    """
    Hyperfine DICOM headers store the full sequence json files for the scan.
    This is stored under two private tags, one for acquisition, one for reconstruction parameters.
    This function extracts these json files and returns them as python dictionaries.

    :param dicom_fname: DICOM filename
    """
    ACQ = "0x03511008"
    RECON = "0x03511009"

    dcm = pyd.dcmread(dicom_fname)

    acq_json = dcm[ACQ].value
    recon_json = dcm[RECON].value

    if isinstance(acq_json, bytes):
        acq_json = acq_json.decode("utf-8")

    if isinstance(recon_json, bytes):
        recon_json = recon_json.decode("utf-8")

    return json.loads(acq_json), json.loads(recon_json)


def g_stejskal_tanner(b_value: float, d_small: float, d_big: float):
    """ 
    Function calculates the gradient magntidude using the 'Stejskal-Tanner' equation (ST-equ).
    Hence, we are assuming a square-wave diffusion sensitisation pulse sequence.

    :param b_value: degree of diffusion sensitisation (s/mm2)
    :param d_small: lowercase delta in ST-equ (s)
    :param d_big: uppercase delta in ST-equ (s)

    Note, the ST-equ is: b = GAMMA^2 * G^2 * d_small^2 * (d_big - d_small/3)
    """

    G_2 = b_value * 1e6 / (GAMMA**2 * d_small**2 * (d_big - d_small / 3))
    G = G_2**0.5

    return G


def dicom_dwi_gradient(dicom_fname: str):
    """
    Calculate the diffusion gradient magnitude for a DWI scan using the Stejskal-Tanner equation
    using the sequence parameters stored in a Hyperfine DICOM header.
    
    :param dicom_fname: DICOM filename
    """

    acq_param_dict, _ = dicom_sequence_params(dicom_fname)
    b = acq_param_dict["app_data"]["b_value_computed"]
    d_small = acq_param_dict["app_data"]["dw_gradient_duration"]
    te = acq_param_dict["timing"]["TE"]
    d_big = te / 2 # This is a approximation but sufficient for our purposes

    return g_stejskal_tanner(b, d_small, d_big)


def dicom_dwi_direction(dicom_fname):
    """
    Extract diffusion direction as a letter (X, Y, Z) from DICOM header.
    """

    dcm = pyd.dcmread(dicom_fname)

    # Check that scan is fact a diffusion scan
    if not dcm.AcquisitionContrast == "DIFFUSION":
        raise ValueError(f"Scan is not a diffusion scan, but has contrast type: {dcm.AcquisitionContrast}")
    
    # Simplest way to extract diffusion direction is from the DICOM header.
    dwi_header = dcm.SharedFunctionalGroupsSequence[0].MRDiffusionSequence[0]
    if dwi_header.DiffusionDirectionality == "DIRECTIONAL":
        dwi_vec = np.array(dwi_header.DiffusionGradientDirectionSequence[0].DiffusionGradientOrientation)

        if np.isclose(np.max(np.abs(dwi_vec)), 1.0, atol=1e-3):
            i_axis = np.argmax(np.abs(dwi_vec))
            direction = "XYZ"[i_axis]
        else:
            # fallback to sequence name
            acq_param_dict, _ = dicom_sequence_params(dicom_fname)

            try:
                direction = acq_param_dict["app_data"]["dwi_axis"].upper()

            except KeyError:
                seq_name = acq_param_dict["app_info"]["sequence_name"].upper()
                direction_list = [a for a in "XYZ" if a in seq_name]

                if len(direction_list) != 1:        
                    raise ValueError("Unclear from scanname the dwi axis of the scan")
    
                direction = direction_list[0]

    # However, for b=0 scans the diffusion direction isn't stored in the DICOM header,
    # instead we can try to extract it from the sequence parameters.
    else:
        acq_param_dict, _ = dicom_sequence_params(dicom_fname)

        try:
            direction = acq_param_dict["app_data"]["dwi_axis"].upper()

        except KeyError:
            seq_name = acq_param_dict["app_info"]["sequence_name"].upper()
            direction_list = [a for a in "XYZ" if a in seq_name]

            if len(direction_list) != 1:        
                raise ValueError("Unclear from scanname the dwi axis of the scan")
    
            direction = direction_list[0]

    return direction


def dicom_scantime(dicom_fname):

    dcm = pyd.dcmread(dicom_fname)
    datetime_string = int(float(dcm.AcquisitionDateTime))

    return datetime_string


def convert_timestamp(datetime_string, datetime_format = "%Y_%m_%d_%H_%M_%S"):
    _datetime_string = str(datetime_string) # Ensure standard python string not numpy string
    return datetime.strptime(_datetime_string, datetime_format)


def dicom_scanname(dicom_fname, display_name=False):
    """
    Extract the scan name fromt the sequence parameters stored in the DICOM header.
    By default, the function will extract the 'sequence name' but can be told to return
    the 'display name' instead which is a more user-friendly name however can be less 
    informative.

    :param dicom_fname: DICOM filename
    :param display_name: If True, extract the 'display name' instead of the 'sequence name' from the DICOM header.
    """

    acq_param_dict, _ = dicom_sequence_params(dicom_fname)

    if display_name:
        name = acq_param_dict["app_info"]["display_name"].replace(" ", "_")
    else:
        name = acq_param_dict["app_info"]["sequence_name"].replace(".user", "")
    return name


def dicom_f0(dicom_fname, hz=True):

    dcm = pyd.dcmread(dicom_fname)
    f0_Mhz = float(dcm.SharedFunctionalGroupsSequence[0].MRImagingModifierSequence[0].TransmitterFrequency)
    
    return f0_Mhz * 1e6 if hz else f0_Mhz


def grad_vec(g: float, axis: str):
    """
    Convert a gradient magnitude and axis into a 3D gradient vector.

    :param g: Gradient magnitude
    :param axis: Axis of the input gradient
    """

    a = axis.upper()

    if a not in "XYZ":
        raise ValueError("Axis must be one of 'X', 'Y', or 'Z'")
    
    i_axis = "XYZ".find(a)
    gvec = np.zeros((3))
    gvec[i_axis] = g

    return gvec


def estimate_temp(scan_dt: datetime, temp_log, time_window_minutes=80, f_past = 1, n_min=None):
    """
    Estimate the temperature of the phantom by taking the mean of temperature readings within a window of time of 
    the scan time. The window is defined by the time_window_minutes and f_past.

    :param scan_dt: datatime object of the scan time of interest
    :param temp_log: list of tuples of (datatime object, temperature reading)
    :param time_window_minutes: duration of time window from which to average temperature readings (minutes)
    :param f_past: fraction of the time window to place with respect to the past of the scan time. """

    if not isinstance(scan_dt, datetime):
        raise ValueError("scan_dt must be a datetime object")

    if not all(isinstance(t_dt, datetime) for t_dt, _ in temp_log) or not all(isinstance(temp, float) for _, temp in temp_log):
        raise ValueError("temp_log must be a list of entries with type (datetime, float)")
    
    MEASUREMENT_INTERVAL = 5 # minutes

    if n_min is None:
        n_min = (time_window_minutes // MEASUREMENT_INTERVAL) - 1

    past_delta = timedelta(minutes=time_window_minutes * f_past)
    future_delta = timedelta(minutes=time_window_minutes * (1 - f_past))
    window_temp_data = [temp for t_dt, temp in temp_log if scan_dt - past_delta <= t_dt <= scan_dt + future_delta]

    if not window_temp_data:
        return None
    elif len(window_temp_data) < n_min:
        return None
    else:
        return float(np.mean(window_temp_data))


def water_diffusivity(temp, polynomial=True):
    
    """
    Calculate the diffusivity of water for the temperature provided to the function.
    :param temp: temperature in degrees Celsius
    :param polynomial: If True, use a second-order polynomial fit to the diffusivity data. If False, use a linear fit.
    
    Returns:
        diffusivity in units: mm2/s
    """

    if polynomial:
        D_um2_ms = 0.0004 * temp **2 + 0.0374 * temp + 1.1222
    else:
        D_um2_ms = 0.0587 * temp + 0.9084
    D_mm2_s = D_um2_ms * 1e-3
    
    return D_mm2_s


def dicom_diffusion_gradient_duration(dicom_fname: str):

    acq_param_dict, _ = dicom_sequence_params(dicom_fname)
    d_small = acq_param_dict["app_data"]["dw_gradient_duration"]
    return d_small


def dicom_echotime(dicom_fname: str):

    dcm = pyd.dcmread(dicom_fname)
    te_ms = dcm["EchoTime"].value # ms
    te = te_ms / 1e3
    return te


def get_mask(image: np.ndarray, lt_percentile=None, mask_type=np.uint8):

    image_a = ants.from_numpy(image)

    if lt_percentile is None:
        mask_a = ants.get_mask(image_a)
    else:
        lt = np.percentile(image, q=lt_percentile)
        mask_a = ants.get_mask(image_a, low_thresh=lt)

    mask = mask_a.numpy().astype(mask_type)

    return mask


def translation_registration(fixed: np.ndarray, moving: np.ndarray, spacing=None):

    if not isinstance(fixed, np.ndarray) or not isinstance(moving, np.ndarray):
        raise ValueError(f"Fixed and moving images must be numpy arrays, but received types: fixed {type(fixed)}, moving {type(moving)}")
    
    if not np.array_equal(fixed.shape, moving.shape):
        raise ValueError(f"Fixed and moving images must have the same shape for registration, but received shapes: fixed {fixed.shape}, moving {moving.shape}")
    
    if spacing is None:
        _spacing = [1] * fixed.ndim
    elif isinstance(spacing, list):
        _spacing = spacing.copy()
    elif isinstance(spacing, (tuple, np.ndarray)):
        _spacing = list(spacing)
    else:
        raise ValueError(f"Spacing must be a list, tuple or numpy array, but received type: {type(spacing)}")
    
    if len(_spacing) != fixed.ndim:
        raise ValueError(f"Spacing must have the same number of dimensions as the input images, but received spacing with {len(_spacing)} dimensions and images with {fixed.ndim} dimensions")

    # Convert numpy arrays to ants images
    fixed_a = ants.from_numpy(fixed, spacing=_spacing)
    moving_a = ants.from_numpy(moving, spacing=_spacing)

    # apply translation registration
    trans = ants.registration(fixed=fixed_a, 
                              moving=moving_a, 
                              type_of_transform="Translation",
                              grad_step=0.1,
                              metric="mattes")
    reg_a = trans["warpedmovout"]
    reg = reg_a.numpy().astype(np.float32)

    return reg


def scaling_registration(fixed: np.ndarray, moving: np.ndarray, spacing=None, scale_tol=0.07):

    if not isinstance(fixed, np.ndarray) or not isinstance(moving, np.ndarray):
        raise ValueError(
            f"Fixed and moving images must be numpy arrays, but received "
            f"fixed {type(fixed)}, moving {type(moving)}"
        )

    if fixed.shape != moving.shape:
        raise ValueError(
            f"Fixed and moving images must have the same shape for registration, "
            f"but received fixed {fixed.shape}, moving {moving.shape}"
        )

    if spacing is None:
        _spacing = [1.0] * fixed.ndim
    elif isinstance(spacing, list):
        _spacing = spacing.copy()
    elif isinstance(spacing, (tuple, np.ndarray)):
        _spacing = list(spacing)
    else:
        raise ValueError(
            f"Spacing must be a list, tuple or numpy array, but received type: {type(spacing)}"
        )

    if len(_spacing) != fixed.ndim:
        raise ValueError(
            f"Spacing must have the same number of dimensions as the input images, "
            f"but received spacing with {len(_spacing)} dimensions and images with {fixed.ndim} dimensions"
        )


    fixed_itk = sitk.GetImageFromArray(fixed.astype(np.float32))
    fixed_itk.SetSpacing(_spacing)

    moving_itk = sitk.GetImageFromArray(moving.astype(np.float32))
    moving_itk.SetSpacing(_spacing)

    fixed_mask = get_mask(fixed).astype(np.uint8)
    fixed_mask_itk = sitk.GetImageFromArray(fixed_mask)
    fixed_mask_itk.SetSpacing(_spacing)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(fixed_mask_itk)
    com_fixed = stats.GetCentroid(1)

    reg_params = [
        {"mv_mask": False, "sampling_factor": 0.2, "mask_lt_percentile": None},
        {"mv_mask": False, "sampling_factor": 1.0, "mask_lt_percentile": None},
        {"mv_mask": True,  "sampling_factor": 1.0, "mask_lt_percentile": None},
        {"mv_mask": True,  "sampling_factor": 1.0, "mask_lt_percentile": 40},
    ]

    last_transform = None
    last_axes_scaling = None
    last_volume_scaling = None
    stabilised = False

    for i, params in enumerate(reg_params):

        registration_method = sitk.ImageRegistrationMethod()

        scale = sitk.ScaleTransform(3)
        scale.SetCenter(com_fixed)
        scale.SetScale([1.0, 1.0, 1.0])
        registration_method.SetInitialTransform(scale, inPlace=False)

        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
        registration_method.SetMetricSamplingPercentage(params["sampling_factor"])

        if params["mv_mask"]:
            moving_mask = get_mask(moving, lt_percentile=params["mask_lt_percentile"]).astype(np.uint8)
            moving_mask_itk = sitk.GetImageFromArray(moving_mask)
            moving_mask_itk.SetSpacing(_spacing)
            registration_method.SetMetricMovingMask(moving_mask_itk)

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-6,
            numberOfIterations=1000,
            gradientMagnitudeTolerance=1e-8,
        )

        transform = registration_method.Execute(fixed_itk, moving_itk)

        base_transform = transform.GetNthTransform(0)
        axes_scaling = np.array(base_transform.GetScale(), dtype=float)
        volume_scaling = float(np.prod(axes_scaling))

        last_transform = transform
        last_axes_scaling = axes_scaling
        last_volume_scaling = volume_scaling

        if abs(volume_scaling - 1.0) <= scale_tol:
            stabilised = True
            break

    if stabilised:
        # print(f"Registration stabilised - Axes scaling = {last_axes_scaling}")
        registered_moving_sitk = sitk.Resample(
            image1=moving_itk,
            referenceImage=fixed_itk,
            transform=last_transform,
            interpolator=sitk.sitkLinear,
            defaultPixelValue=0.0,
            outputPixelType=moving_itk.GetPixelID(),
        )

        reg_moving = sitk.GetArrayFromImage(registered_moving_sitk) * last_volume_scaling
        return reg_moving, stabilised

    else:
        print("Failed to stabilise registration, returning original moving image")
        return moving, stabilised


def register_to_b0(data_4d: np.ndarray, bvals: np.ndarray, spacing=None, b0_threshold=5):

    # Load b-values and identify the index of the first b=0 scan to use as the fixed image for registration
    i0_b0 = np.where(bvals < b0_threshold)[0][0]
    fixed = data_4d[...,i0_b0]

    registered_vols = []
    i_dropped = []
    for i in tqdm(range(bvals.shape[0]), desc="Registering DWI images to b=0 image", mininterval=0.1):
        moving = data_4d[..., i]

        if i == i0_b0:
            registered_vols.append(moving)  # Fixed image goes unmodified
            continue

        reg_moving_1 = translation_registration(fixed, moving, spacing)
        reg_moving_2, stablised = scaling_registration(fixed, reg_moving_1, spacing)
        
        if stablised:
            registered_vols.append(reg_moving_2)
        else:
            i_dropped.append(i)

    # --- Reassemble into 4D volume and save ---
    registered_data = np.stack(registered_vols, axis=-1)
    print("Registration complete")

    return registered_data, np.array(i_dropped)


def process_dicom(dicom_fname):

    scanname = dicom_scanname(dicom_fname)

    scantime_string_format = "%Y%m%d%H%M%S"
    scantime = dicom_scantime(dicom_fname)
    scantime_dt = convert_timestamp(scantime, scantime_string_format)
    scan_te = dicom_echotime(dicom_fname)

    bval = dicom_bvalue(dicom_fname)
    acq_dict, recon_dict = dicom_sequence_params(dicom_fname)
    dwi_axis = dicom_dwi_direction(dicom_fname)
    dwi_grad = dicom_dwi_gradient(dicom_fname)
    dwi_gvec = grad_vec(dwi_grad, dwi_axis)
    dwi_delta = dicom_diffusion_gradient_duration(dicom_fname)

    scan_nii = dcm2niix_wrapper(dicom_fname)
    img = scan_nii.get_fdata()
    aff = scan_nii.affine
    spacing = spacing_from_affine(aff)

    outputs = {"seq_name": scanname,
               "scan_time": scantime_dt,
               "dwi_direction": dwi_axis,
               "b_value": bval,
               "g_value": dwi_grad,
               "grad_vec": dwi_gvec,
               "dwi_grad_duration": dwi_delta,
               "image": img,
               "affine": aff,
               "spacing": spacing,
               "TE": scan_te}
    
    return outputs


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


def M_ss(t1, TR=0.85, theta=np.pi/2):
    return np.sin(theta) * ((1-np.exp(-TR/t1))/(1-np.cos(theta) * np.exp(-TR/t1)))