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
from models.Unet import UNet3D
from models.modules import ShareRefiner_Layer, PromptRefer
from monai.data import MetaTensor
class CAT(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, args, backbone = 'swinunetr'):
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
        elif backbone == 'unet':
            self.backbone = UNet3D()
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))
        
        self.ap_emb = torch.load(args.anatomical_prompt_path, map_location='cpu')#json.load(open(args.anatomical_prompts_paths))
        self.tp_emb = torch.load(args.textual_prompt_path, map_location='cpu')
        self.only_last = args.only_last
        if not self.only_last:
            in_dims = [768, 768, 384, 192]
            dims = [768, 384, 192, 96]
        else:
            in_dims = [768]
            dims = [768]
            
        AttendLayers = []
        
        for i in range(len(dims)):
            layer = ShareRefiner_Layer(dim=dims[i], out_dim=dims[i], num_heads=8, 
                    norm_layer=nn.LayerNorm, 
                    in_features=in_dims[i], 
                    mlp_ratio=4, 
                    hard=True, 
                    gumbel=True, 
                    sum_assign=False,
                    ap_project=True,
                    assign_eps=1., 
                    gumbel_tau=1.)
            AttendLayers.append(layer)
        self.attend_layers = nn.ModuleList(AttendLayers)
        
        self.seg_query = nn.Embedding(out_channels, dims[0])
        
        
        self.prompt_refer = PromptRefer(dim=dims[-1], out_dim=dims[-1], num_heads=8, norm_layer=nn.LayerNorm, in_features=dims[-1], mlp_ratio=4, )
        
        self.controller = nn.Linear(dims[-1], 48)
        self.out_norm_layer = nn.LayerNorm(48)
        
        #-------norm----------
        self.ap_projection = nn.Linear(dims[-1], dims[-1])
        self.tp_projection = nn.Linear(dims[-1], dims[-1])
        self.query_projection = nn.Linear(dims[-1], dims[-1])
         
        
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')


    def forward(self, x_in, text_query):
        B = x_in.shape[0]
        out, feats = self.backbone(x_in)
        #print(x_in.device)
        seg_query = self.seg_query.weight
        seg_query = seg_query.unsqueeze(0).repeat(B, 1, 1)

        ap_emb = self.ap_emb.to(x_in.device)
        batch_ap_emb = ap_emb.unsqueeze(0).repeat(B, 1, 1).to(x_in.device)

        tp_emb = self.tp_emb.to(x_in.device)
        batch_tp_emb = tp_emb.unsqueeze(0).repeat(B, 1, 1).to(x_in.device)
        
        for i in range(len(self.attend_layers)):
            seg_query = self.attend_layers[i](feats[i], seg_query=seg_query, anatomical_prompts=None, is_ap=False, is_seg=True)
            batch_tp_emb = self.attend_layers[i](feats[i], textual_prompts=batch_tp_emb, anatomical_prompts=None, is_ap=False)
            batch_ap_emb = self.attend_layers[i](feats[i], textual_prompts=None, anatomical_prompts=batch_ap_emb, is_ap=True)
        
        N = seg_query.shape[1]    
        batch_p_emb = torch.cat([batch_ap_emb, batch_tp_emb], dim=1)

        
        seg_query = self.prompt_refer(seg_query, batch_p_emb, None)
        
        weight = self.out_norm_layer(self.controller(seg_query))
        
        B, C, D, H, W = out.size()
        logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ weight.transpose(1, 2)
        logits_out = logits.transpose(1, 2).reshape(B, N, D, H, W) 
        
        proj_s = F.normalize(self.tp_projection(tp_emb), dim=-1)
        proj_tp = F.normalize(self.tp_projection(text_query), dim=-1)
        
        sim_s2i = proj_s @ proj_tp.T
        max_index = torch.argmax(sim_s2i).item()
        
        return logits_out[:,max_index:max_index+1,:]