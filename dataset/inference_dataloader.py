import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd,
    CropForegroundd, ToTensord, MapTransform
)
from monai.config import KeysCollection
from monai.utils import MetaKeys
from monai.data import MetaTensor
import torch
class ToMetaTensord(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], torch.Tensor) and not isinstance(d[key], MetaTensor):
                d[key] = MetaTensor(d[key])
        return d

def get_pred_transforms(args):
    pred_transforms = Compose(
        [
            LoadImaged(keys=["image"]),  # 用自定义的 LoadNPZd 替换 LoadImaged
            ToMetaTensord(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
        ]
    )
    return pred_transforms