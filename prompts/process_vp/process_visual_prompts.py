import json
import os

volumes = []

for root, _, files in os.walk('/SegFM3D_DATA_TR_split_volumes'):
    for file in files:
        if file.endswith('.pt'):
            volumes.append(os.path.join(root, file))

with open(f"volumes_path.json", "w") as f:
    json.dump(volumes, f, indent=4, ensure_ascii=False, separators=(',', ': '))


import json
import torch
import os
import random
from transformers import BertModel
from transformers import AutoTokenizer

save_dir = "vp_feats"
os.makedirs(save_dir, exist_ok=True)

Part_Names = json.load(open('total_classes.json'))

volumes_feats = []

for root, _, files in os.walk('volumes_feats'):
    for file in files:
        if file.endswith('.pt'):
            volumes_feats.append(os.path.join(root, file))

feats_map = {}
for volumes_feat in volumes_feats:
    part_name = volumes_feat.split('/')[-2]
    assert part_name in Part_Names
    if part_name not in feats_map:
        feats_map[part_name] = [volumes_feat]
    else:
        feats_map[part_name].append(volumes_feat)
    
vp_data = []
    
for i in range(1000):
    vps = []
    for part in Part_Names:
        vp_item = random.choice(feats_map[part])
        vp_feat = torch.load(vp_item)
        vps.append(vp_feat)
    file_name = 'vprompt_{:05d}.pt'.format(i)
    vis_emb = torch.cat(vps, dim=0)
    save_path = os.path.join(save_dir, file_name)
    torch.save(vis_emb, save_path) 