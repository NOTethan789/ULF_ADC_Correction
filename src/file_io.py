"""
File handling helpers for scraping data directories and saving NIfTI outputs.
"""

import os
from glob import glob
import nibabel as nib


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


def save_nifti(nifti, dir_path: str, filename: str):
    save_path = os.path.join(dir_path, filename)
    nib.save(nifti, save_path)
    print(f"{filename} saved")
