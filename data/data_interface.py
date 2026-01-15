import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random

class TrainCollater:
    def __init__(self,
                 llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        inputs_text = [sample['sources'] for sample in batch]
        targets_text = [sample['targets'] + self.terminator for sample in batch]
        poi_embeds = [sample['poi_embeds'] for sample in batch]
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p < thresh_hold or not self.train:
           flag = False
        else:
           flag = True
        #flag = True
        self.cur_step += 1
        if self.train:
            # targets_text = [target_text + self.terminator for target_text in targets_text]
            inputs_pair = [[p,t] for p, t in zip(inputs_text, targets_text)]
            batch_tokens = self.llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                # padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            new_batch = {
                "tokens":batch_tokens,
                "poi_embeds":poi_embeds,
                "flag":flag
                }
        else:
            # targets_text = [target_text + self.terminator for target_text in targets_text]
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                # padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            new_batch = {
                "tokens":batch_tokens,
                "poi_embeds":poi_embeds,
                "answer": targets_text,
                "flag": flag
            }
        return new_batch

class DInterface(pl.LightningDataModule):

    def __init__(self, 
                 llm_tokenizer=None,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.llm_tokenizer=llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        # self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.data_module(dataset=self.dataset, stage='train')
        self.valset = self.data_module(dataset=self.dataset, stage='val')
        self.testset = self.data_module(dataset=self.dataset,  stage='test')
        self.max_steps = self.max_epochs*(len(self.trainset)//self.batch_size)//self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(llm_tokenizer=self.llm_tokenizer, train=True, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(llm_tokenizer=self.llm_tokenizer, train=False))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(llm_tokenizer=self.llm_tokenizer, train=False))

    def load_data_module(self):
        name = "poi_data"
        camel_name = "PoiData"
        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

