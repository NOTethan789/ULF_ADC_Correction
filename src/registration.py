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


def _resolve_spacing(spacing, ndim):

    if spacing is None:
        _spacing = [1.0] * ndim
    elif isinstance(spacing, list):
        _spacing = spacing.copy()
    elif isinstance(spacing, (tuple, np.ndarray)):
        _spacing = list(spacing)
    else:
        raise ValueError(f"Spacing must be a list, tuple or numpy array, but received type: {type(spacing)}")

    if len(_spacing) != ndim:
        raise ValueError(f"Spacing must have the same number of dimensions as the input image, but received spacing with {len(_spacing)} dimensions and image with {ndim} dimensions")

    return _spacing


def translation_registration(fixed: np.ndarray, moving: np.ndarray, spacing=None, return_transform=False, type_of_transform="Translation"):
    """
    Register 'moving' to 'fixed' with a translation (ANTs, Mattes MI). Pass type_of_transform="Rigid" for a
    rigid (rotation + translation) registration, see rigid_registration.

    Returns the registered image, or (registered image, transform) if return_transform is True.
    The transform is a dict of the ANTs affine parameters, which can be re-applied to another
    image on the same grid with apply_translation.
    """

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
                              type_of_transform=type_of_transform,
                              grad_step=0.1,
                              metric="mattes")
    reg_a = trans["warpedmovout"]
    reg = reg_a.numpy().astype(np.float32)

    if return_transform:
        tx = ants.read_transform(trans["fwdtransforms"][0])
        transform = {"parameters": np.asarray(tx.parameters, dtype=float).tolist(),
                     "fixed_parameters": np.asarray(tx.fixed_parameters, dtype=float).tolist()}
        return reg, transform

    return reg


def rigid_registration(fixed: np.ndarray, moving: np.ndarray, spacing=None, return_transform=False):
    """
    Register 'moving' to 'fixed' with a rigid transform (rotation + translation; ANTs, Mattes MI).
    The returned transform is applied with apply_translation, like a translation transform.
    """
    return translation_registration(fixed, moving, spacing, return_transform, type_of_transform="Rigid")


def rotation_angles_deg(transform: dict):
    """Rotation angles (degrees, about the x, y and z axes) of a transform from translation_registration or rigid_registration."""
    R = np.asarray(transform["parameters"][:9], dtype=float).reshape(3, 3)
    ax = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    ay = np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))
    az = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return [float(ax), float(ay), float(az)]


def apply_translation(image: np.ndarray, transform: dict, spacing=None):
    """
    Apply a transform returned by translation_registration or rigid_registration to another image on the same grid.
    """
    _spacing = _resolve_spacing(spacing, image.ndim)
    image_a = ants.from_numpy(image, spacing=_spacing)

    tx = ants.create_ants_transform(transform_type="AffineTransform",
                                    dimension=image.ndim,
                                    parameters=np.asarray(transform["parameters"]),
                                    fixed_parameters=np.asarray(transform["fixed_parameters"]))

    reg_a = ants.apply_ants_transform_to_image(tx, image_a, image_a, interpolation="linear")
    return reg_a.numpy().astype(np.float32)


def scaling_registration(fixed: np.ndarray, moving: np.ndarray, spacing=None, scale_tol=0.07, return_transform=False):
    """
    Register 'moving' to 'fixed' with an anisotropic scaling about the centroid of the fixed mask (SimpleITK).

    Returns (registered image, stabilised), or (registered image, stabilised, transform) if return_transform
    is True. The transform is None if registration failed to stabilise; otherwise it is a dict that can be
    re-applied to another image on the same grid with apply_scaling.
    """

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

        if return_transform:
            transform = {"centre": [float(c) for c in com_fixed],
                         "scale": last_axes_scaling.tolist(),
                         "volume_scaling": last_volume_scaling}
            return reg_moving, stabilised, transform

        return reg_moving, stabilised

    else:
        print("Failed to stabilise registration, returning original moving image")

        if return_transform:
            return moving, stabilised, None

        return moving, stabilised


def apply_scaling(image: np.ndarray, transform: dict, spacing=None):
    """
    Apply a transform returned by scaling_registration to another image on the same grid.
    Includes the same volume-scaling intensity correction as scaling_registration.
    """
    _spacing = _resolve_spacing(spacing, image.ndim)

    image_itk = sitk.GetImageFromArray(image.astype(np.float32))
    image_itk.SetSpacing(_spacing)

    scale = sitk.ScaleTransform(3)
    scale.SetCenter(transform["centre"])
    scale.SetScale(transform["scale"])

    registered_sitk = sitk.Resample(
        image1=image_itk,
        referenceImage=image_itk,
        transform=scale,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        outputPixelType=image_itk.GetPixelID(),
    )

    return sitk.GetArrayFromImage(registered_sitk) * transform["volume_scaling"]


def register_to_b0(data_4d: np.ndarray, bvals: np.ndarray, spacing=None, b0_threshold=5, return_transforms=False, first_transform="translation"):
    """
    Register every volume in data_4d to the first b=0 volume (translation, then scaling). With
    first_transform="rigid" the first step is a rigid (rotation + translation) registration instead.

    Returns (registered_data, i_dropped), where i_dropped are the indices of volumes that failed to register
    and are excluded from registered_data. If return_transforms is True, also returns a list with one entry
    per input volume: {"translation": ..., "scaling": ...} for registered volumes, {"fixed": True} for the
    reference volume and None for dropped volumes. Pass this list to apply_registration to register another
    4D array (e.g. the noisy scans) in exactly the same way.
    """

    if first_transform not in ("translation", "rigid"):
        raise ValueError(f"first_transform must be 'translation' or 'rigid', received {first_transform}")
    first_registration = rigid_registration if first_transform == "rigid" else translation_registration

    # Load b-values and identify the index of the first b=0 scan to use as the fixed image for registration
    i0_b0 = np.where(bvals < b0_threshold)[0][0]
    fixed = data_4d[...,i0_b0]

    registered_vols = []
    i_dropped = []
    transforms = []
    for i in tqdm(range(bvals.shape[0]), desc="Registering DWI images to b=0 image", mininterval=0.1):
        moving = data_4d[..., i]

        if i == i0_b0:
            registered_vols.append(moving)  # Fixed image goes unmodified
            transforms.append({"fixed": True})
            continue

        reg_moving_1, translation = first_registration(fixed, moving, spacing, return_transform=True)
        reg_moving_2, stablised, scaling = scaling_registration(fixed, reg_moving_1, spacing, return_transform=True)
        
        if stablised:
            registered_vols.append(reg_moving_2)
            transforms.append({"translation": translation, "scaling": scaling})
        else:
            i_dropped.append(i)
            transforms.append(None)

    # --- Reassemble into 4D volume and save ---
    registered_data = np.stack(registered_vols, axis=-1)
    print("Registration complete")

    if return_transforms:
        return registered_data, np.array(i_dropped), transforms

    return registered_data, np.array(i_dropped)


def register_pairs_to_b0(data_4d: np.ndarray, bvals: np.ndarray, directions, spacing=None, b0_threshold=5, return_transforms=False, b0_transform="translation", reference="first"):
    """
    Paired registration for protocols that acquire a b=0 scan and then a diffusion-weighted scan for each
    direction. The volumes of data_4d must be in acquisition order.

    1. Every b=0 volume is registered to a reference b=0 volume (same contrast, so robust). b0_transform is
       "translation" or "rigid" (rotation + translation). reference is "first" (the first b=0 volume),
       "middle" (the b=0 volume whose position is most central, i.e. the smallest largest translation to the
       other b=0 volumes, found with a quick rigid registration of all b=0 volumes to the first) or the index
       of a b=0 volume in data_4d.
    2. A diffusion-weighted volume is paired with the most recent preceding b=0 volume of the same direction.
       It receives the translation of its b=0 partner (the two were acquired close together, so the head
       will have moved by about the same amount).
    3. A scaling registration of the translated diffusion-weighted volume onto its translated b=0 partner then
       corrects the distortion that is specific to the diffusion gradients.

    Returns (registered_data, i_dropped) where i_dropped are volumes whose scaling registration failed to
    stabilise and are excluded from registered_data. If return_transforms is True, a list with one entry per
    input volume is also returned, in the format of register_to_b0 (the "translation" and "scaling" entries
    are None where not applied), so it can be passed to apply_registration.
    """

    bvals = np.asarray(bvals)
    directions = np.asarray(directions)
    is_b0 = bvals < b0_threshold
    if not is_b0.any():
        raise ValueError("No b=0 volume to register to")
    if b0_transform not in ("translation", "rigid"):
        raise ValueError(f"b0_transform must be 'translation' or 'rigid', received {b0_transform}")
    b0_registration = rigid_registration if b0_transform == "rigid" else translation_registration

    i_b0 = np.where(is_b0)[0]
    if reference == "first":
        i_ref = int(i_b0[0])
    elif reference == "middle":
        # Position of every b=0 volume relative to the first one, then pick the volume whose largest
        # distance to the others is smallest
        shifts = np.zeros((i_b0.size, 3))
        for k, i in enumerate(i_b0[1:], start=1):
            _, tr = rigid_registration(data_4d[..., i_b0[0]], data_4d[..., i], spacing, return_transform=True)
            shifts[k] = tr["parameters"][-3:]
        dist = np.linalg.norm(shifts[:, None, :] - shifts[None, :, :], axis=-1)
        i_ref = int(i_b0[np.argmin(dist.max(axis=1))])
    elif isinstance(reference, (int, np.integer)) and reference in i_b0:
        i_ref = int(reference)
    else:
        raise ValueError(f"reference must be 'first', 'middle' or the index of a b=0 volume, received {reference}")
    print(f"Reference b=0 volume: {i_ref}")
    fixed = data_4d[..., i_ref]

    n_vols = bvals.shape[0]
    registered = [None] * n_vols
    transforms = [None] * n_vols
    translations = {}
    i_dropped = []

    # --- 1. registration of the b=0 volumes ---
    for i in tqdm(np.where(is_b0)[0], desc=f"{b0_transform.capitalize()} registration of b=0 volumes", mininterval=0.1):
        if i == i_ref:
            registered[i] = data_4d[..., i]
            transforms[i] = {"fixed": True}
            continue
        reg, translation = b0_registration(fixed, data_4d[..., i], spacing, return_transform=True)
        registered[i] = reg
        translations[i] = translation
        transforms[i] = {"translation": translation, "scaling": None}

    # --- 2./3. pair every diffusion-weighted volume with its preceding b=0 and register it ---
    for i in tqdm(np.where(~is_b0)[0], desc="Scaling registration of DWI volumes to their b=0", mininterval=0.1):
        partners = [j for j in np.where(is_b0)[0] if j < i and directions[j] == directions[i]]
        if not partners:
            raise ValueError(f"Volume {i} (direction {directions[i]}) has no preceding b=0 volume of the same direction")
        j = partners[-1]

        translation = translations.get(j)   # None if the partner is the reference volume
        moving = data_4d[..., i]
        if translation is not None:
            moving = apply_translation(moving, translation, spacing)

        reg, stabilised, scaling = scaling_registration(registered[j], moving, spacing, return_transform=True)
        if stabilised:
            registered[i] = reg
            transforms[i] = {"translation": translation, "scaling": scaling}
        else:
            i_dropped.append(i)

    registered_data = np.stack([registered[i] for i in range(n_vols) if transforms[i] is not None], axis=-1)
    print("Paired registration complete")

    if return_transforms:
        return registered_data, np.array(i_dropped), transforms

    return registered_data, np.array(i_dropped)


def apply_registration(data_4d: np.ndarray, transforms: list, spacing=None):
    """
    Apply the transforms from register_to_b0(..., return_transforms=True) to another 4D array with the
    same grid and volume order. Volumes that were dropped during registration are dropped here too, so
    the output matches the registered data from register_to_b0.
    """

    if data_4d.shape[-1] != len(transforms):
        raise ValueError(f"data_4d has {data_4d.shape[-1]} volumes but {len(transforms)} transforms were supplied")

    registered_vols = []
    for i, transform in enumerate(tqdm(transforms, desc="Applying registration transforms", mininterval=0.1)):

        if transform is None:
            continue  # Volume was dropped during registration

        moving = data_4d[..., i]

        if "fixed" in transform:
            registered_vols.append(moving)  # Reference image goes unmodified
            continue

        reg_moving = moving
        if transform.get("translation") is not None:
            reg_moving = apply_translation(reg_moving, transform["translation"], spacing)
        if transform.get("scaling") is not None:
            reg_moving = apply_scaling(reg_moving, transform["scaling"], spacing)
        registered_vols.append(reg_moving)

    return np.stack(registered_vols, axis=-1)
