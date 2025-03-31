import os
import json
import h5py
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp
from tqdm import tqdm

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
)
from monai.data import DataLoader, Dataset, list_data_collate
from monai.data.meta_tensor import MetaTensor
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_to_numpy, convert_to_tensor
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import torch.multiprocessing as mp_torch
mp_torch.set_sharing_strategy('file_system')

from utils.utils import get_key

ORGAN_DATASET_DIR = 'SegFM3D_DATA_TR'
ORGAN_LIST = './datalist/SegFM3D_DATA_train.json'
NUM_CLASS = 170
TRANSFER_LIST = []
TEMPLATE = json.load(open('utils/TEMPLATE.json'))

# --------------------------
# 1. 定义自定义的 transforms
# --------------------------
class ToTemplatelabel(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, totemplate: List, tumor=False, tumor_list=None) -> NdarrayOrTensor:
        new_lbl = np.zeros(lbl.shape, dtype=np.float32)
        for src, tgt in enumerate(totemplate):
            new_lbl[lbl == (src + 1)] = tgt
        if tumor:
            for src, item in tumor_list:
                new_lbl[new_lbl == item] = totemplate[0]
        return new_lbl

class ToTemplatelabeld(MapTransform):
    '''
    Comment: spleen to 1
    '''
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.totemplate = ToTemplatelabel()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        template_key = d['name'].split('_')[0]
        # 可以视需要决定是否加 tumor 标记
        TUMOR = False
        tumor_list = None
        d['label'] = self.totemplate(d['label'], TEMPLATE[template_key], tumor=TUMOR, tumor_list=tumor_list)
        return d

def generate_label(input_lbl, num_classes, name, TEMPLATE, raw_lbl):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input_lbl: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    Comment: spleen to 0
    """
    shape = np.array(input_lbl.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    input_lbl = input_lbl.long()

    B = result.shape[0]
    for b in range(B):
        template_key = name[b].split('_')[0]
        organ_list = TEMPLATE[template_key]
        # -1 for organ not labeled
        for i in range(num_classes):
            if (i + 1) not in organ_list:
                result[b, i] = -1
            else:
                result[b, i] = (input_lbl[b][0] == (i + 1))
    return result


# --------------------------
# 2. 定义核心处理函数 worker
# --------------------------
def worker(sub_data_dicts):
    """
    每个进程都要做的工作：对分到的那一部分 data_dict 做 transform 并存储结果。
    """
    # 这里为了避免 DataLoader 的多进程与外部多进程冲突，可以把 num_workers 设成 0 或 1
    label_process = Compose(
        [
            LoadImaged(keys=["label", "label_raw"]),#"image", 
            EnsureChannelFirstd(keys=["label", "label_raw"]),
            Orientationd(keys=["label", "label_raw"], axcodes="RAS"),
            Spacingd(
                keys=["label", "label_raw"],
                pixdim=(1.5, 1.5),
                mode=("nearest", "nearest"),
                align_corners=True
            ),
            ToTemplatelabeld(keys=['label']),
        ]
    )

    local_dataset = Dataset(data=sub_data_dicts, transform=label_process)
    local_loader = DataLoader(local_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=list_data_collate)

    for index, batch in enumerate(local_loader):
        y, y_raw, name = batch["label"], batch['label_raw'], batch['name']#batch["image"], 
        y = generate_label(y, NUM_CLASS, name, TEMPLATE, y_raw)   # [1, num_classes, D, H, W]
        name_str = batch['name'][0].replace('label', 'post_label')
        post_dir = os.path.join(ORGAN_DATASET_DIR, *name_str.split('/')[:-1])
        store_y = y.numpy().astype(np.uint8)
        if not os.path.exists(post_dir):
            os.makedirs(post_dir, exist_ok=True)
        save_path = os.path.join(ORGAN_DATASET_DIR, f"{name_str}.h5")
        print(save_path)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)


# --------------------------
# 3. 主进程：把数据分块并行处理
# --------------------------
if __name__ == "__main__":
    # 准备你的 data_dicts_train
    train_img = []
    train_lbl = []
    train_name = []

    # 读取 JSON，拿到原始列表
    all_data = json.load(open(ORGAN_LIST))
    for each in all_data:
        key = get_key(each["img"])
        if len(TRANSFER_LIST) == 0 or key in TRANSFER_LIST:
            train_img.append(os.path.join(ORGAN_DATASET_DIR, each["img"]))
            train_lbl.append(os.path.join(ORGAN_DATASET_DIR, each["label"]))
            train_name.append(each["label"].split('.')[0])

    data_dicts_train = [
        {'image': image, 'label': label, 'label_raw': label, 'name': name}
        for image, label, name in zip(train_img, train_lbl, train_name)
    ]
    print('total train len {}'.format(len(data_dicts_train)))

    # 自行决定要并行的进程数 n_processes
    n_processes = 64

    # 将 data_dicts_train 分成 n_processes 份
    chunk_size = len(data_dicts_train) // n_processes
    sub_datasets = []
    for i in range(n_processes):
        start = i * chunk_size
        # 最后一个进程把剩余都拿走
        end = len(data_dicts_train) if i == n_processes - 1 else (i+1) * chunk_size
        sub_datasets.append(data_dicts_train[start:end])

    # 利用进程池并行处理
    with mp.Pool(n_processes) as pool:
        # 如果想在主进程加一个进度条，可以这样写：
        list(tqdm(pool.imap_unordered(worker, sub_datasets), total=n_processes))
        # 或者简单一点：pool.map(worker, sub_datasets)
        # 当 worker 中本身也带 tqdm 时，需要小心多进程下 tqdm 的互相干扰，
        # 建议只在主进程做一个大的进度提示即可（如上）
    
    print("All processes done!")
