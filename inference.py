import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import monai
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# 如果没有就需要安装: pip install transformers
from transformers import AutoTokenizer, BertModel

from models.CAT_inference import CAT
from dataset.inference_dataloader import get_pred_transforms
from utils import loss
from utils.utils import dice_score, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS, extract_topk_largest_candidates
import SimpleITK as sitk

from monai.transforms import Resize, Compose
from monai.transforms import Invertd, SaveImaged, SaveImage
from monai.data.meta_tensor import MetaTensor

torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', help='device')

    # logging
    parser.add_argument('--log_name', default='CAT_test', help='log name')
    parser.add_argument('--save_dir', default='CAT_pred', help='save dir')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    # model load
    parser.add_argument('--pretrain_weights', 
                        default='infer_weights/weights.pth', 
                        help='path to pretrained checkpoint')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    # dataset
    parser.add_argument('--anatomical_prompt_paths', default='prompts/vprompts.json', help='visual_prompt_path')
    parser.add_argument('--textual_prompt_paths', default='prompts/vprompts.json', help='text_prompt_path')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='num workers for DataLoader')
    parser.add_argument('--a_min', default=0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=255, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument("--only_last", action="store_false", help="only atten last feat")
    parser.add_argument('--threshold', default=0.5, type=float, help='binarization threshold')

    # 额外可选：如果只是推理少量图像，可以不需要 dataset_list，而是直接传入图像路径
    parser.add_argument('--single_infer_path', default='', 
                        help='if set, only do inference for a single Nifti file path')

    args = parser.parse_args()
    return args

def npz2nii(npz_path):
    item = os.path.basename(npz_path)
    save_img_dir = os.path.join('tmp_img')
    os.makedirs(save_img_dir, exist_ok=True)
    save_img_name = os.path.join(save_img_dir, item.replace('.npz', '.nii.gz'))

    npz = np.load(npz_path, allow_pickle=True)
    imgs = npz['imgs']  # Image data
    # Convert NumPy arrays to SimpleITK images
    img_sitk = sitk.GetImageFromArray(imgs)
    spacing = npz['spacing']
    try:
        if len(spacing) != 3:
            raise ValueError("Spacing Error")
        img_sitk.SetSpacing(tuple(spacing))
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        pass
    
    # Save the image NIfTI
    sitk.WriteImage(img_sitk, save_img_name)
    
    text_prompts = npz['text_prompts'].item()
    text_prompts.pop('instance_label', None)

    return save_img_name, text_prompts
            
def load_model(args):
    """
    加载 CAT 模型，并载入预训练权重
    """
    # 准备 3D 模型
    model = CAT(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=NUM_CLASS,
        backbone=args.backbone,
        args=args
    )
    # 加载预训练权重
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
    load_dict = checkpoint['net']
    # 也可能需要同时恢复 epoch 等信息
    # args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        # 如果你的 checkpoint 中 key 是 "module.xxx" 这种多卡训练形式，则可做进一步处理
        name = '.'.join(key.split('.')[1:])  # 这里是根据你的实际保存格式来
        if name in store_dict:
            store_dict[name] = value

    model.load_state_dict(store_dict, strict=True)
    print('Use pretrained weights from:', args.pretrain_weights)
    model.eval()
    model.to(args.device)

    return model


def load_text_encoder(args):
    """
    加载文本编码器（BioBERT）和 tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("biobert-base-cased-v1.2")
    text_encoder = BertModel.from_pretrained("biobert-base-cased-v1.2").to(args.device)
    text_encoder.eval()
    return text_encoder, tokenizer


def model_wrapper(model, text_emb):
    """
    由于使用 MONAI 原生 sliding_window_inference 时，其默认只传入图像，
    因此我们需要把 text_emb "固定" 在模型里，返回一个只接受图像作为输入的函数。
    """
    def wrapped_inference(input_image):
        # 这里确保 model.forward 可以同时处理图像和文本向量
        return model(input_image, text_query=text_emb)
    return wrapped_inference


def visualize_label(batch_list, save_dir, name, input_transform):
    ### function: save the prediction result into dir
    ## Input
    ## batch: the batch dict output from the monai dataloader
    ## one_channel_label: the predicted reuslt with same shape as label
    ## save_dir: the directory for saving
    ## input_transform: the dataloader transform
    
    for i in batch_list:
        i['meta_dict'] = i['image'].meta
        i['one_channel_label_v1'] = MetaTensor(i['one_channel_label_v1'])
        i['meta_dict']['filename_or_obj'] = name
        
        img_meta = i['image']
        assert isinstance(img_meta, MetaTensor), "image must be MetaTensor"
        assert len(img_meta.shape) == 4, f"MetaTensor should be [C, D, H, W], got shape {img_meta.shape}"
    
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


def inference_single_image(img_path, text_prompts, model, text_encoder, tokenizer, pred_transforms, args):
    """
    对单张 3D 图像做推理，并可视化结果
    """
    # 根据当前权重文件名，构造保存目录
    test_item = args.pretrain_weights.split('/')[-1].split('.')[0]  # e.g., CAT_weights_part
    save_path = os.path.join(args.save_dir, test_item)
    pred_save_path = os.path.join(save_path, 'predict')
    os.makedirs(pred_save_path, exist_ok=True)

    # 1) 构造用于推理的 data_dict
    data_dicts_test = [{'image': img_path, 'name': os.path.basename(img_path).split('.')[0]}]
    
    # 2) 对 data_dict 进行预处理 transforms
    #    如果你的 pred_transforms 是一个 Compose，需要这样： 
    processed_data = [pred_transforms(d) for d in data_dicts_test]

    
    target_ids = []
    texts = []
    
    for label in text_prompts:
        target_ids.append(int(label))
        texts.append(text_prompts[label])
    
    # 3) 利用文本编码器获得 text_emb
    device = torch.device(args.device)
    text_tokens = tokenizer(texts, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=150, 
                            return_tensors="pt").to(device)
    text_output = text_encoder(
        text_tokens.input_ids, 
        attention_mask=text_tokens.attention_mask, 
        return_dict=True
    )
    text_emb = text_output.last_hidden_state[:, 0, :]  # [batch=1, hidden_dim]

    # 4) 包装模型，使之兼容 MONAI 的 sliding_window_inference
    slice_model = model_wrapper(model, text_emb)

    # 5) 推理
    for index, batch_data in enumerate(processed_data):
        # batch_data["image"] 应该是 [C, D, H, W] 或 [1, D, H, W] 之类
        # MONAI sliding_window_inference 需要 [B, C, D, H, W]
        image = batch_data["image"].as_tensor().unsqueeze(0).to(device)  # 增加 batch 维度
        name = batch_data["name"]

        with torch.no_grad():
            pred = sliding_window_inference(
                inputs=image, 
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                sw_batch_size=1,
                predictor=slice_model,
                overlap=0.5,
                mode='gaussian'
            )
            pred_sigmoid = F.sigmoid(pred)

        # 6) 阈值化
        # pred_sigmoid shape: [B=1, out_channels=NUM_CLASS, D, H, W]
        B, outC, D, H, W = pred_sigmoid.shape
        threshold_list = torch.tensor([args.threshold]).repeat(B, outC, 1, 1, 1).to(device)
        pred_hard = (pred_sigmoid > threshold_list).float()[0]  # [1, NUM_CLASS, D, H, W]
        
        pred_hard = pred_hard.cpu().numpy()

        post_pred_mask = np.zeros(pred_hard.shape)
        for x in range(post_pred_mask.shape[0]):
            post_pred_mask[x] = extract_topk_largest_candidates(pred_hard[x], 1)
        pred_hard_post = torch.tensor(post_pred_mask)
        assert pred_hard_post.shape[0] == len(target_ids), f"{pred_hard_post.shape[0]}, {len(target_ids)}"
        merged_label_v1 = torch.zeros(1, D, H, W)
        for x, target_id in enumerate(target_ids):
            merged_label_v1[0][pred_hard_post[x]==1] = target_id
        batch_data['one_channel_label_v1'] = merged_label_v1
        batch_data['image'] = processed_data[index]["image"]
        
        item_name = name.split('/')[-1].split('.')[0]
        
        visualize_label([batch_data], pred_save_path, item_name, pred_transforms)
        
        nii_path = os.path.join(pred_save_path, f"{item_name}.nii.gz")
        img = sitk.ReadImage(nii_path)
        arr = sitk.GetArrayFromImage(img)
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{item_name}.npz")
        arr_uint8 = arr.astype(np.uint8)
        np.savez_compressed(output_path, segs=arr_uint8)
        
        torch.cuda.empty_cache()
        
        
        
def main():
    args = get_args()
    
    # 1) 加载模型
    model = load_model(args)

    # 2) 加载文本编码器和 tokenizer
    text_encoder, tokenizer = load_text_encoder(args)
    
    pred_transforms = get_pred_transforms(args)

    # 如果只想对单个图像做推理
    if args.single_infer_path:
        if not args.single_infer_path.endswith('.npz'):
            raise ValueError("The input file must be a .npz file.")
        if not os.path.exists(args.single_infer_path):
            raise FileNotFoundError(f"File not found: {args.single_infer_path}")
        single_infer_path, text_prompts = npz2nii(args.single_infer_path)
        inference_single_image(single_infer_path, text_prompts, model, text_encoder, tokenizer, pred_transforms, args)
        return

if __name__ == "__main__":
    main()
