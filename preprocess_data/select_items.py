import os
import nibabel as nib
import numpy as np
import json

total_classes = json.load(open('info_files/total_classes.json'))

dataset_class2id = json.load(open('info_files/dataset_class2id.json'))

dataset_map = json.load(open('info_files/dataset_map.json'))

img_root = 'SegFM3D_DATA_TR_ALL'

split_mask_root = 'SegFM3D_DATA_TR_split_masks'


mask_label_map = {}

for index_dataset in os.listdir(img_root):
    index_label_dir = os.path.join(img_root, index_dataset, 'label')
    save_mask_dir = os.path.join(split_mask_root, index_dataset)
    for label_item in os.listdir(index_label_dir):
        label_path = os.path.join(index_label_dir, label_item)
        if label_item.endswith('.nii.gz') and not label_item.startswith('.'):
            sub_name = label_item.split('.')[0]
            save_mask_sub_dir = os.path.join(save_mask_dir, sub_name)
            for key, value in dataset_class2id[dataset_map[index_dataset]].items():
                mask_name = f"{key}.nii.gz"
                mask_path = os.path.join(save_mask_sub_dir, mask_name)
                mask_label_map[json.dumps([mask_path, key])] = label_path

with open('info_files/mask_img_map.json', 'w') as f:
    json.dump(mask_label_map, f, indent=4, separators=(',', ': '), ensure_ascii=False)


mask_img_map = json.load(open('info_files/mask_img_map.json'))

part_org = {}

for combination_str in mask_img_map:
    combination = json.loads(combination_str)
    part_name = combination[1]#mask_path.split('/')[-1].split('.')[0]
    if part_name not in part_org:
        part_org[part_name] = [mask_img_map[combination_str]]
    else:
        part_org[part_name].append(mask_img_map[combination_str])

for item in total_classes:
    if item not in part_org:
        print(item) 

import random
selected_part = {}
for part_name in part_org:
    if len(part_org[part_name]) > 10:
        selected_part[part_name] = random.sample(part_org[part_name], 10)
    else:
        selected_part[part_name] = part_org[part_name]

with open('info_files/selected_part.json', 'w') as f:
    json.dump(selected_part, f, indent=4, separators=(',', ': '), ensure_ascii=False)


import os
import json
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count

# 读取必要的映射文件
selected_part = json.load(open('info_files/selected_part.json'))
dataset_class2id = json.load(open('info_files/dataset_class2id.json'))
dataset_map = json.load(open('info_files/dataset_map.json'))

img_root = 'SegFM3D_DATA_TR_ALL'

def extract_organ_masks(label_nii_path):
    """提取 NIFTI 文件中的器官标签信息"""
    label_nii = sitk.ReadImage(label_nii_path)
    label_data = sitk.GetArrayFromImage(label_nii)
    
    # 获取所有唯一标签值，排除0（背景）
    unique_labels = np.unique(label_data)
    max_value = np.max(unique_labels)
    return int(max_value), unique_labels

def process_dataset(index_dataset):
    """处理单个数据集目录，检查是否需要修复 mask"""
    index_label_dir = os.path.join(img_root, index_dataset, 'label')
    if not os.path.exists(index_label_dir):
        return None

    dataset_name = dataset_map.get(index_dataset)
    if dataset_name is None or dataset_name not in dataset_class2id:
        return None

    ids = [int(id) for id in dataset_class2id[dataset_name].values()]
    max_id = max(ids)

    for label_item in os.listdir(index_label_dir)[:2]:
        label_path = os.path.join(index_label_dir, label_item)
        if label_item.endswith('.nii.gz') and not label_item.startswith('.'):
            o_max_id, unique_labels = extract_organ_masks(label_path)
            if max_id < o_max_id:
                print(dataset_name, ids, unique_labels)
                print("=====================================")
                return index_dataset  # 发现问题即返回

    return None  # 没有问题返回 None



if __name__ == '__main__':
    # 获取数据集目录列表
    index_datasets = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]

    # 使用多进程池处理
    with Pool(processes=64) as pool:
        results = pool.map(process_dataset, index_datasets)

    # 过滤掉 None 结果
    masks_need_fix = [res for res in results if res is not None]

    # 保存结果
    with open('info_files/masks_need_fix.json', 'w') as f:
        json.dump(masks_need_fix, f, indent=4, separators=(',', ': '), ensure_ascii=False)

    