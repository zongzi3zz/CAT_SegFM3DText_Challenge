import os, sys
import cc3d
import fastremap
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pyod.models.knn import KNN
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from scipy import ndimage

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize, Compose
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

from monai.data import decollate_batch
from monai.transforms import Invertd, SaveImaged
import json
NUM_CLASS = 192



TEMPLATE = json.load(open('utils/TEMPLATE.json'))

ORGAN_NAME = json.load(open('utils/total_classes.json'))

assert len(ORGAN_NAME) == NUM_CLASS

## mapping to original setting
MERGE_MAPPING_v1 = json.load(open('utils/MERGE_MAPPING_v1.json'))



def organ_post_process(pred_mask, organ_list, save_dir, dataset_id, case_id, args):
    post_pred_mask = np.zeros(pred_mask.shape)
    plot_save_path = save_dir
    log_path = args.log_name
    # dataset_id = save_dir.split('/')[-2]
    # case_id = save_dir.split('/')[-1]
    if not os.path.isdir(plot_save_path) and args.store_result:
        os.makedirs(plot_save_path)
    for b in range(pred_mask.shape[0]):
        for x in range(pred_mask.shape[1]):
            post_pred_mask[b,x] = pred_mask[b,x]
    return post_pred_mask


def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()
    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label


def threshold_organ(data, organ=None, threshold=None):
    ### threshold the sigmoid value to hard label
    ## data: sigmoid value
    ## threshold_list: a list of organ threshold
    B = data.shape[0]
    N = data.shape[1]
    threshold_list = []
    for i in range(N):
        threshold_list.append(0.5)
    threshold_list = torch.tensor(threshold_list).repeat(B, 1).reshape(B,len(threshold_list),1,1,1).cuda()
    pred_hard = data > threshold_list
    return pred_hard


def visualize_label(batch, save_dir, name, input_transform):
    ### function: save the prediction result into dir
    ## Input
    ## batch: the batch dict output from the monai dataloader
    ## one_channel_label: the predicted reuslt with same shape as label
    ## save_dir: the directory for saving
    ## input_transform: the dataloader transform
    
    batch_list = decollate_batch(batch)
    for i in batch_list:
        i['meta_dict'] = i['image'].meta
        i['one_channel_label_v1'] = MetaTensor(i['one_channel_label_v1'])
        i['meta_dict']['filename_or_obj'] = name
    
    post_transforms = Compose([
        Invertd(
            keys=['one_channel_label_v1'], #, 'split_label'
            transform=input_transform,
            orig_keys="image",
            meta_keys="meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        SaveImaged(keys='one_channel_label_v1',
                meta_keys = 'meta_dict',
                output_postfix="",
                output_dir=save_dir, 
                separate_folder = False,
                resample=False
        ),
    ])
    
    batch = [post_transforms(i) for i in batch_list]

def merge_label(pred_bmask, name):
    B, C, W, H, D = pred_bmask.shape
    merged_label_v1 = torch.zeros(B,1,W,H,D).cuda()
    #merged_label_v2 = torch.zeros(B,1,W,H,D).cuda()
    
    for b in range(B):
        template_key = get_key(name[b])
        transfer_mapping_v1 = MERGE_MAPPING_v1[template_key]
        #transfer_mapping_v2 = MERGE_MAPPING_v2[template_key]
        organ_index = []
        for item in transfer_mapping_v1:
            src, tgt = item
            merged_label_v1[b][0][pred_bmask[b][src-1]==1] = tgt
        # for item in transfer_mapping_v2:
        #     src, tgt = item
        #     merged_label_v2[b][0][pred_bmask[b][src-1]==1] = tgt
            # organ_index.append(src-1)
        # organ_index = torch.tensor(organ_index).cuda()
        # predicted_prob = pred_sigmoid[b][organ_index]
    return merged_label_v1#, merged_label_v2


def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name.split('_')[0])
    template_key = name.split('_')[0]
    return template_key


def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)


    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

if __name__ == "__main__":
    threshold_organ(torch.zeros(1,12,1))    
