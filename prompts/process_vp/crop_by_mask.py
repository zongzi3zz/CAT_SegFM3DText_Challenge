import os
import json
import nibabel as nib
import numpy as np
import multiprocessing as mp
import cc3d

# 读取JSON配置文件
selected_part = json.load(open('info_files/selected_part.json'))
dataset_class2id = json.load(open('info_files/dataset_class2id.json'))
dataset_map = json.load(open('info_files/dataset_map.json'))

output_root = 'SegFM3D_DATA_TR_split_masks'

def get_largest_connected_component(mask_data):
    labels_out, num_features = cc3d.connected_components(mask_data, return_N=True)
    if num_features == 0:
        return np.zeros_like(mask_data)
    
    component_sizes = np.bincount(labels_out.ravel())[1:]  # 忽略背景 (label 0)
    largest_component_label = np.argmax(component_sizes) + 1
    
    return (labels_out == largest_component_label).astype(mask_data.dtype)

def process_image(args):
    img_path, mask_path, output_path = args
    # 创建输出文件夹
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return
    
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    
    try:
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
    except:
        print(f"Error loading {mask_path}")
        return
    
    # 获取最大连通分量
    largest_cc_mask = get_largest_connected_component(mask_data)
    
    coords = np.array(np.where(largest_cc_mask > 0))
    if coords.size == 0:
        print(f"No foreground found in {mask_path}")
        return
    
    mask_min_coords = coords.min(axis=1)
    mask_max_coords = coords.max(axis=1)
    center_coords = np.mean(coords, axis=1).astype(int)
    
    half_size = np.array([48, 48, 48])
    min_coords = center_coords - half_size
    max_coords = center_coords + half_size
    
    # 确保裁剪范围在图像内
    min_coords = np.maximum(np.minimum(min_coords, mask_min_coords), [0, 0, 0])
    max_coords = np.minimum(np.maximum(max_coords, mask_max_coords), np.array(img_data.shape) - 1)
    
    roi_data = img_data[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]
    
    new_affine = img.affine.copy()
    new_affine[:3, 3] += min_coords * new_affine[:3, :3].diagonal()
    
    roi_img = nib.Nifti1Image(roi_data, new_affine, img.header)
    nib.save(roi_img, output_path)
    
    print(f'Processed {img_path}, ROI has been saved to {output_path}')



if __name__ == "__main__":
    num_workers = min(mp.cpu_count(), 64)  # 限制最大进程数，避免过载
    pool = mp.Pool(processes=num_workers)

    tasks = []
    for part_name in selected_part:
        for label_path in selected_part[part_name]:
            mask_path = os.path.join(output_root, part_name, os.path.basename(label_path))
            output_path = mask_path.replace('SegFM3D_DATA_TR_split_masks', 'SegFM3D_DATA_TR_split_volumes')
            if '31_CT_LymphNode' in label_path:
                image_path = label_path.replace('/label/', '/img/')
                tasks.append((image_path, mask_path, output_path))

    # 并行执行任务
    pool.map(process_image, tasks)
    
    # 关闭进程池
    pool.close()
    pool.join()