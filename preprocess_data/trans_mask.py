import SimpleITK as sitk
import numpy as np
import os
import json
import multiprocessing
from tqdm import tqdm
import shutil
def trans_data(args):
    """
    读取 NIFTI 标签文件，将所有 label >= 1 的区域转换为同一类，并保存为新的 NIFTI 文件。
    
    :param args: (label_nii_path, output_path) 元组
    """
    label_nii_path, output_path = args
    # shutil.copy(output_path, label_nii_path)
    
    try:
        # 加载 NIFTI 文件
        label_nii = sitk.ReadImage(label_nii_path)
        label_data = sitk.GetArrayFromImage(label_nii)
        
        # 获取原始数据类型
        original_dtype = label_data.dtype
        
        # 创建二值掩码：所有 label >= 1 的像素设为 1，否则设为 0，并保持原始数据类型
        binary_mask = (label_data >= 1).astype(original_dtype)
        
        # 创建新的 SimpleITK 图像
        new_nii = sitk.GetImageFromArray(binary_mask)
        new_nii.CopyInformation(label_nii)  # 保持原始的空间信息
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(label_nii_path, output_path)
        # 保存处理后的 NIFTI 文件
        sitk.WriteImage(new_nii, label_nii_path)
    except Exception as e:
        print(f"Error processing {label_nii_path}: {e}")

def process_dataset(index_dataset, img_root, tmp_root, masks_need_fix):
    """ 处理单个数据集中的所有 NIFTI 标签文件 """
    if index_dataset not in masks_need_fix:
        return []
    index_label_dir = os.path.join(img_root, index_dataset, 'label')
    if not os.path.exists(index_label_dir):
        return []
    
    output_label_dir = os.path.join(tmp_root, index_dataset, 'label')
    os.makedirs(output_label_dir, exist_ok=True)
    
    tasks = []
    for label_item in os.listdir(index_label_dir):
        label_path = os.path.join(index_label_dir, label_item)
        output_path = os.path.join(output_label_dir, label_item)
        if label_item.endswith('.nii.gz') and not label_item.startswith('.'):
            tasks.append((label_path, output_path))
    
    return tasks

def main():
    masks_need_fix = json.load(open('info_files/masks_need_fix.json'))
    img_root = 'SegFM3D_DATA_TR'
    tmp_root = 'SegFM3D_DATA_TMP_LABEL'
    
    all_tasks = []
    for index_dataset in os.listdir(img_root):
        tasks = process_dataset(index_dataset, img_root, tmp_root, masks_need_fix)
        if tasks:
            all_tasks.extend(tasks)
    
    # 使用 multiprocessing 进行加速，并显示进度条
    with multiprocessing.Pool(processes=64) as pool:
        list(tqdm(pool.imap_unordered(trans_data, all_tasks), total=len(all_tasks), desc="Processing NIFTI Files"))
    
if __name__ == "__main__":
    main()
