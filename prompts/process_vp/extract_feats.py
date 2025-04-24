import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import json
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from models.Extractor import Extractor
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
torch.multiprocessing.set_sharing_strategy('file_system')


def process(args):        
    device = torch.device(args.device)
    # prepare the 3D model
    model = Extractor(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding=None,
                    args=args
                    )

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])
    
    
    visual_prompts_paths = json.load(open('volumes_path.json'))
    model.to(device=device)
    model.eval()
    from tqdm import tqdm
    with torch.no_grad():
        for volumes_path in tqdm(visual_prompts_paths):
            vp_emb = model.extract_feats(volumes_path)
            save_path = volumes_path.replace('SegFM3D_DATA_TR_split_volumes', 'volumes_feats')
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(vp_emb, save_path)

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    ## logging
    parser.add_argument('--log_name', default='tumor_kgvit', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt',  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
