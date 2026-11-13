"""
Image masking, resampling and registration (translation / scaling) helpers.
"""

import numpy as np
from tqdm import tqdm
import ants
import SimpleITK as sitk


def spacing_from_affine(affine:np.ndarray):
    R = affine[0:3, 0:3] # Spacing & orientation subarray
    spacing = np.linalg.norm(R, axis=0)
    return spacing.tolist()


def get_mask(image: np.ndarray, lt_percentile=None, mask_type=np.uint8):

    image_a = ants.from_numpy(image)

    if lt_percentile is None:
        mask_a = ants.get_mask(image_a)
    else:
        lt = np.percentile(image, q=lt_percentile)
        mask_a = ants.get_mask(image_a, low_thresh=lt)

    mask = mask_a.numpy().astype(mask_type)

    return mask


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
