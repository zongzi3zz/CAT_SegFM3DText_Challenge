import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm
from monai.transforms.io.array import LoadImage
from monai.transforms import Compose, EnsureChannelFirst, Orientation, Spacing, ScaleIntensityRange, CropForeground, SpatialPad, Resize, ToTensor


def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files


def process_file(args):
    """
    Process a single NIfTI file.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    None
    """
    file_path, transforms = args
    img_data = transforms(file_path)
    file_name = os.path.basename(file_path)
    output_dir = os.path.dirname(file_path)

    file_name = file_name.split(".")[0]+".pt"
    save_path = os.path.join(output_dir, file_name)
    torch.save(img_data, save_path)

# Example usage:
if __name__ == "__main__":
    split_to_preprocess = 'SegFM3D_DATA_TR_split_volumes' #select the validation or test split
    nii_files = read_nii_files(split_to_preprocess)
    transforms = Compose(
        [
            LoadImage(), #0
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRange(
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForeground(),
            SpatialPad(spatial_size=(96, 96, 96), mode='constant'),
            Resize(spatial_size=(96, 96, 96)),
            ToTensor(),            
        ]
    )
    file_transforms = [(file, transforms) for file in nii_files]
    num_workers = 128  # Number of worker processes

    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, file_transforms), total=len(nii_files)))
    
