from typing import Sequence, Tuple, Type, Union
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from models.SwinUNETR import SwinUNETR
from monai.data import MetaTensor
class Extractor(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, args, backbone = 'swinunetr', encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))
        
        
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        else:
            raise Exception(f'{self.backbone_name} is not implemented in curretn version')

    def extract_feats(self, path):
        visual_img = torch.load(path, map_location='cuda').unsqueeze(0)
        vp_feats = self.backbone.swinViT(visual_img)[-1]
        vp_emb = vp_feats.flatten(2,4).transpose(1,2).mean(dim=1)
        print(vp_emb.size())
        return vp_emb


