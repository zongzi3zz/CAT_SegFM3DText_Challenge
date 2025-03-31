import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from models.CAT_pred import CAT
from dataset.dataloader_Seg import get_test_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ
import gc

import random

torch.multiprocessing.set_sharing_strategy('file_system')


def model_wrapper(model, ap_emb, tp_emb, organ_list):
    """
    由于使用 MONAI 原生 sliding_window_inference 时，其默认只传入图像，
    因此我们需要把 text_emb "固定" 在模型里，返回一个只接受图像作为输入的函数。
    """
    def wrapped_inference(input_image):
        # 这里确保 model.forward 可以同时处理图像和文本向量
        return model(input_image, ap_emb=ap_emb, tp_emb=tp_emb, organ_list=organ_list)
    return wrapped_inference


def validation(model, ValLoader, val_transforms, args):
    test_item = args.pretrain_weights.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_dir, test_item)
    pred_save_path = os.path.join(save_path,'predict') 
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)
    model.eval()
    dice_list = {}
    if args.store_result:
        cases = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    import json
    anatomical_prompts_paths = json.load(open(args.anatomical_prompts_paths))
    textual_prompts_paths = json.load(open(args.text_prompt_path))

    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name, o_label = batch["image"].cuda(), batch["post_label"].as_tensor().float(), batch["name"], batch['label']
        ap_emb = torch.load(random.choice(anatomical_prompts_paths), map_location='cpu').to(args.device)
        tp_emb = torch.load(random.choice(textual_prompts_paths), map_location='cpu').to(args.device)
        template_key = get_key(name[0])
        organ_list = TEMPLATE[template_key]
        slice_model = model_wrapper(model, ap_emb, tp_emb, organ_list)
        try:
            with torch.no_grad():
                pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, slice_model, overlap=0.5, mode='gaussian')# sw_device=torch.device('cuda'), device=torch.device('cpu'), buffer_steps=2
                pred_sigmoid = F.sigmoid(pred)
        except Exception as e:
            print(f"{e}", image.shape)
            continue
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        del ap_emb
        del tp_emb
        torch.cuda.empty_cache()
        gc.collect()

        B = pred_hard.shape[0]
        for b in range(B):
            if args.store_result:
                cases[name[b]] = {}
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_path = os.path.join(pred_save_path, name[b].split('/')[0])
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, pred_path, dataset_id = name[b].split('/')[0], case_id = name[b].split('/')[-1], args = args)
            pred_hard_post = torch.tensor(pred_hard_post)
            index = 0
            for organ in organ_list:
                if organ > 0:
                    if args.store_result:
                        dice_organ, recall, precision, spe_sen = dice_score(pred_hard_post[b,index,:,:,:].cuda(), label[b,index,:,:,:].cuda(), spe_sen = args.store_result)
                    else:
                        dice_organ, recall, precision = dice_score(pred_hard_post[b,index,:,:,:].cuda(), label[b,index,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                    if args.store_result:
                        cases[name[b]][ORGAN_NAME[organ-1]] = [dice_organ.item(), recall.item(), precision.item(), spe_sen.item()]
                    index+=1
        torch.cuda.empty_cache()
    if args.store_result:
        import json
        file_name = '+'.join(args.dataset_list)
        with open(os.path.join(save_path, f'{file_name}_cases.json'), 'w') as f:
            json.dump(cases, f, indent=4, ensure_ascii=False, separators=(',', ': '))
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    file_name = '+'.join(args.dataset_list)
    with open(os.path.join(save_path, f'{file_name}_test_record.txt'), 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        print(content)
        f.write(content)
        f.write('\n')
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='cuda')
    ## logging
    parser.add_argument('--log_name', default='CAT_test', help='The path resume from checkpoint')
    parser.add_argument('--save_dir', default='test_res', help='save dir')
    ## model load
    parser.add_argument('--pretrain_weights', default='', help='The path resume from checkpoint')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['SegFM3D_DATA'])

    parser.add_argument('--data_root_path', default='SegFM3D_DATA_VAL', help='data root path')
    parser.add_argument('--data_file_path', default='./datalist/', help='data txt path')
    parser.add_argument('--anatomical_prompts_paths', default='./prompts/vprompts.json', help='visual_prompt_path')
    parser.add_argument('--text_prompt_path', default='./prompts/tprompts.json', help='text_prompt_path')
    
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=255, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument("--only_last", action="store_false", help="only atten last feat")
    parser.add_argument('--threshold', default=0.5, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = CAT(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    args=args
                    )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain_weights)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict, strict=False)
    print('Use pretrained weights')

    model.to('cuda')

    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_test_loader(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
