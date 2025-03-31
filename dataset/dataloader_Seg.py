from monai.transforms import (
    Compose,
    CropForegroundd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    EnsureChannelFirstd,
    SpatialPadd,
    FgBgToIndicesd
)
import os
import json
import sys
from copy import copy, deepcopy
import h5py

import torch.distributed as dist
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
from utils.utils import get_key

from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate

from monai import config
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor, PathLike
from monai.data import SmartCacheDataset, partition_dataset, DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.data.utils import collate_meta_tensor, pickle_operations, dev_collate
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from monai.data import PersistentDataset
import random
from utils.utils import TEMPLATE
DEFAULT_POST_FIX = PostFix.meta()

def o_list_data_collate(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    valid_keys = ['image', 'label', 'post_label', 'name']
    tmp_data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    data = []
    for item in tmp_data:
        tmp_item = {}
        for k in valid_keys:
            # if k == "post_label":
            #     assert item["post_label"].is_sparse
            #     tmp_item[k] = item[k].to_dense()
            # else:
            tmp_item[k] = item[k]
        data.append(tmp_item)
    key = None
    if config.USE_META_DICT:
        data = pickle_operations(data)  # bc 0.9.0
    if isinstance(elem, Mapping):
        ret = {}
        for k in valid_keys:
            key = k
            data_for_batch = [d[key] for d in data]
            ret[key] = collate_meta_tensor(data_for_batch)
    else:
        ret = collate_meta_tensor(data)
    return ret


class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def to_sparse_tensor(self, data, organ_list):
        data = torch.tensor(data, dtype=torch.uint8)
        _, D, W, H = data.shape
        organ_list = torch.as_tensor(organ_list)  # 转换为 PyTorch Tensor
        num_nonzero = (organ_list != 0).sum().item()
        sparse_label = torch.empty((num_nonzero, D, W, H), dtype=torch.uint8)
        index = 0
        for i, organ in enumerate(organ_list):
            if organ == 0:
                continue
            else:
                sparse_label[index] = data[organ - 1]
                index += 1
        return sparse_label
    
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        
        d['post_label'] = data[0]
        # 处理稀疏张量转换
        name = d['name']
        template_key = get_key(name)
        organ_list = TEMPLATE[template_key]
        d['post_label'] = self.to_sparse_tensor(data[0], organ_list)
    
        return d


def get_loader(args):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            FgBgToIndicesd(
                keys="label",  # 计算 label 的前景和背景索引
                fg_postfix="_fg",  # 生成的前景索引 key 为 "label_fg"
                bg_postfix="_bg",  # 生成的背景索引 key 为 "label_bg"
            ),
            
            RandCropByPosNegLabeld(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=0,
                num_samples=args.num_samples,
                fg_indices_key="label_fg",  # 使用 `FgBgToIndicesd` 计算的前景索引
                bg_indices_key="label_bg",  # 使用 `FgBgToIndicesd` 计算的背景索引
                allow_smaller=True,
            ), # 8
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.20,
            # ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []

    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_train.json')
        data = json.load(open(json_path))
        
        for each in data:
            name = each["label"].split('.nii.gz')[0]
            train_img.append(os.path.join(args.data_root_path, each["img"]))
            train_lbl.append(os.path.join(args.data_root_path, each["label"]))
            train_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            train_name.append(name)
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_val.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["label"].split('.')[0]
            val_img.append(os.path.join(args.data_root_path, each["img"]))
            val_lbl.append(os.path.join(args.data_root_path, each["label"]))
            val_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))


    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_test.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["label"].split('.')[0]
            test_img.append(os.path.join(args.data_root_path, each["img"]))
            test_lbl.append(os.path.join(args.data_root_path, each["label"]))
            test_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    if args.phase == 'train':
        data_part = partition_dataset(
            data=data_dicts_train,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=True,
        )[dist.get_rank()]
        if args.cache_dataset:
            train_dataset = CacheDataset(data=data_part, transform=train_transforms, cache_rate=args.cache_rate, num_workers=8)
        else:
            train_dataset = Dataset(data=data_part, transform=train_transforms)
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                    collate_fn=o_list_data_collate)
        return train_loader, train_sampler
        '''
        # If you want to use PersistentDataset to cache data in disk, you can use the following code
        if args.cache_dataset:
            cache_dir = "data_cache"
            data_part = partition_dataset(
                data=data_dicts_train,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]
            
            
            partitioned_dataset = PersistentDataset(
                data=data_part,
                transform=train_transforms,
                cache_dir=cache_dir,
            )

            # 预热写缓存：只处理本 rank 的数据
            loader_phase1 = DataLoader(partitioned_dataset, batch_size=1, num_workers=1, shuffle=False)
            for _ in loader_phase1:
                pass
            print(f"Rank 0: 训练数据缓存完成, 缓存目录: {cache_dir}")

            dist.barrier()
            if args.local_rank == 0:
                print("阶段1并行缓存写入完成!")

            # 然后用同一个 cache_dir，直接读缓存文件
            train_dataset = PersistentDataset(
                data=data_dicts_train,
                transform=train_transforms,
                cache_dir=cache_dir,
            )
            print(f"[Rank={args.local_rank}] 使用磁盘缓存目录: {cache_dir}")
        else:
            train_dataset = Dataset(data=data_part, transform=train_transforms)
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=o_list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler
        '''
    if args.phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms

def get_test_loader(args):
    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )    
    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for item in args.dataset_list:
        json_path = os.path.join(args.data_file_path, item+'_test.json')
        data = json.load(open(json_path))
    
        for each in data:
            name = each["label"].split('.')[0]
            test_img.append(os.path.join(args.data_root_path, each["img"]))
            test_lbl.append(os.path.join(args.data_root_path, each["label"]))
            test_post_lbl.append(os.path.join(args.data_root_path, name.replace('label', 'post_label') + '.h5'))
            test_name.append(name)
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))
    test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return test_loader, val_transforms
