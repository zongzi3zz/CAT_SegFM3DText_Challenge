import numpy as np
import SimpleITK as sitk
import os
from multiprocessing import Pool, cpu_count
import json

dataset_map =  {v: k for k, v in json.load(open('info_files/dataset_map.json'))}


def process_dataset(args):
    original_dir, modal, dataset, save_dataset_name, root = args
    modal_dir = os.path.join(original_dir, modal)

    save_dataset_dir = os.path.join(root, save_dataset_name)
    
    dataset_dir = os.path.join(modal_dir, dataset)
    for item in os.listdir(dataset_dir):
        if item.endswith('.npz') and not item.startswith('.'):
            item_path = os.path.join(dataset_dir, item)
            npz = np.load(item_path, allow_pickle=True)
            imgs = npz['imgs']  # Image data
            gts = npz['gts']    # Ground truth data

            # Convert NumPy arrays to SimpleITK images
            img_sitk = sitk.GetImageFromArray(imgs)
            gt_sitk = sitk.GetImageFromArray(gts)
            spacing = npz['spacing']
            try:
                if len(spacing) != 3:
                    raise ValueError("Spacing 应该包含三个值，分别对应于体素的尺寸。")
                img_sitk.SetSpacing(tuple(spacing))
                gt_sitk.SetSpacing(tuple(spacing))
            except Exception as e:
                print(f"Error processing {item_path}: {e}")
                pass
            
            # Save the image NIfTI
            save_img_dir = os.path.join(save_dataset_dir, 'img')
            os.makedirs(save_img_dir, exist_ok=True)
            save_img_name = os.path.join(save_img_dir, item.replace('.npz', '.nii.gz'))
            sitk.WriteImage(img_sitk, save_img_name)
            
            # Save the ground truth NIfTI
            save_label_dir = os.path.join(save_dataset_dir, 'label')
            os.makedirs(save_label_dir, exist_ok=True)
            save_label_name = os.path.join(save_label_dir, item.replace('.npz', '.nii.gz'))
            sitk.WriteImage(gt_sitk, save_label_name)

def main():
    root = '/Volumes/HSSD/SegFM3D_DATA_TR/'
    os.makedirs(root, exist_ok=True)

    original_dir = '/Volumes/HSSD/SegFM3D/3D_train_npz_random_10percent_16G'
    index = 1
    tasks = []
    
    for modal in os.listdir(original_dir):
        modal_dir = os.path.join(original_dir, modal)
        for dataset in os.listdir(modal_dir):
            assert dataset in dataset_map, f"{dataset} not in dataset_map"
            tasks.append((original_dir, modal, dataset, dataset_map[dataset], root))
            index += 1
    # Number of processes to use (can set this to any number you want)
    num_processes = 64
    # Create a pool of processes and distribute the work
    with Pool(processes=num_processes) as pool:
        pool.map(process_dataset, tasks)

if __name__ == '__main__':
    main()
