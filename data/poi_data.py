import json
import re

import torch
import torch.utils.data as data

def get_poi_embed(texts, embedding):
    all_poi_embeds = []
    for text in texts:
        pattern = r'(At.*?\[PH\])'
        sentences = re.findall(pattern, text)
        user_poi_embeds = []
        for sentence in sentences:
            pattern = r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user_(\d+) visited POI_(\d+) \[PH\]'
            match = re.search(pattern, sentence)
            if match:
                poi_id = int(match.group(3))
                user_poi_embeds.append(embedding[poi_id].to('cpu'))
        all_poi_embeds.append(user_poi_embeds)
    return all_poi_embeds


class PoiData(data.Dataset):
    def __init__(self, dataset, stage=None):
        if stage == 'train':
            data_path = f'data/ref/{dataset}/rakd_train.json'
        else:
            data_path = f'data/ref/{dataset}/rakd_test.json'
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, falling back to original logic.")
            data_path = f'data/ref/{dataset}50/{stage}_qa_pairs_kqt.json'

        with open(data_path, 'r') as f:
            data = json.load(f)

        self.sources = []
        self.targets = []

        for item in data:
            if 'question' in item and 'answer' in item:
                self.sources.append(item['question'])
                self.targets.append(item['answer'])
            else:
                split_index = item['answer'].find('I_') + len('I_')
                self.sources.append(item['question'] + item['answer'][:split_index])
                self.targets.append(item['answer'][split_index:])
        poi_embed = torch.load(f'poi_embed/poi_emb_{dataset}.pth')
        self.poi_embeds = [[] for _ in range(len(self.sources))]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(sources=self.sources[i], targets=self.targets[i], poi_embeds=self.poi_embeds[i])