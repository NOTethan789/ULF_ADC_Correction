"""
Load and parse DICOM files: scan metadata, DWI parameters, conversion to NIfTI.
"""

import numpy as np
import json
from pathlib import Path
import subprocess
import nibabel as nib
import pydicom as pyd
import tempfile
import shutil
from datetime import datetime
import registration as reg

# gyromagnetic ratio of the proton in radians per second per Tesla
GAMMA = 2.675221900e8


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


def dicom_diffusion_gradient_duration(dicom_fname: str):

    acq_param_dict, _ = dicom_sequence_params(dicom_fname)
    d_small = acq_param_dict["app_data"]["dw_gradient_duration"]
    return d_small


def dicom_echotime(dicom_fname: str):

    dcm = pyd.dcmread(dicom_fname)
    te_ms = dcm["EchoTime"].value # ms
    te = te_ms / 1e3
    return te


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
    spacing = reg.spacing_from_affine(aff)

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
