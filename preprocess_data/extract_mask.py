import os
import json
import nibabel as nib
import numpy as np
import multiprocessing as mp

# 读取JSON配置文件
selected_part = json.load(open('info_files/selected_part.json'))
dataset_class2id = json.load(open('info_files/dataset_class2id.json'))
dataset_map = json.load(open('info_files/dataset_map.json'))

# 目标输出目录
output_root = 'SegFM3D_DATA_TR_split_masks'
os.makedirs(output_root, exist_ok=True)

def extract_organ_masks(args):
    try:
        label_nii_path, part_name, output_path = args
        if os.path.exists(output_path):
            #print(f"Mask for label {part_name} already exists at {output_path}")
            return
        # 加载NIFTI文件
        label_nii = nib.load(label_nii_path)
        label_data = label_nii.get_fdata()

        # 获取所有唯一标签值，排除0（背景）
        unique_labels = np.unique(label_data)
        output_folder = os.path.dirname(output_path)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        index_dataset = label_nii_path.split('/')[-3]
        if '_' in part_name:
            part_name = part_name.replace('_', '/')
        part_value = np.dtype(unique_labels.dtype).type(dataset_class2id[dataset_map[index_dataset]][part_name])

        organ_mask = np.zeros_like(label_data)
        
        # 将匹配的标签值设置为1
        closest_value = unique_labels[np.argmin(np.abs(unique_labels - part_value))]
        organ_mask[label_data == closest_value] = 1

        # 保存为新的NIFTI文件
        organ_mask_nii = nib.Nifti1Image(organ_mask, label_nii.affine)
        nib.save(organ_mask_nii, output_path)
        print(f"Saved mask for label {int(part_value)} to {output_path}")
    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    num_workers = 64  # 限制最大进程数，避免过载
    pool = mp.Pool(processes=num_workers)

    tasks = []
    for part_name in selected_part:
        for label_path in selected_part[part_name]:
            output_path = os.path.join(output_root, part_name, os.path.basename(label_path))
            tasks.append((label_path, part_name, output_path))

    # 并行执行任务
    pool.map(extract_organ_masks, tasks)
    
    # # 关闭进程池
    # pool.close()
    # pool.join()

