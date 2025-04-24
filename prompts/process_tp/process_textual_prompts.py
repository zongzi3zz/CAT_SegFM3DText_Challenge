import json
import torch
import os
import random
from transformers import BertModel
from transformers import AutoTokenizer
save_dir = "tp_feats"
os.makedirs(save_dir, exist_ok=True)
device = 'cuda'
text_encoder = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
text_encoder.eval()


Part_Names = json.load(open('total_classes.json'))
organ_prompts = json.load(open('text_prompts.json'))
    
tp_data = []
    
for i in range(1000):
    tps = []
    for part in Part_Names:
        tp_item = random.choice(organ_prompts[part])
        tps.append(tp_item)
    tp_data.append(tps)

with open(f"select_text_prompts.json", "w") as f:
    json.dump(tp_data, f, indent=4, ensure_ascii=False, separators=(',', ': '))

tp_data = json.load(open("select_text_prompts.json"))

for i, item in enumerate(tp_data):
    with torch.no_grad():
        text_tokens = tokenizer(item, padding='max_length', truncation=True, max_length=150, return_tensors="pt").to(device)
        text_output = text_encoder(text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True)
        text_emb = text_output.last_hidden_state[:, 0, :].detach().cpu()
        assert text_emb.shape[0] == 170, f"{text_emb.shape}"
        file_name = 'tprompt_{:05d}.pt'.format(i)
        save_path = os.path.join(save_dir, file_name)
        torch.save(text_emb, save_path) 
