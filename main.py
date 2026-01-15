import os
import pytorch_lightning as pl
from argparse import ArgumentParser
import json
from tqdm import tqdm
import math

import torch
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model.model_interface import MInterface
from data.data_interface import DInterface
from model.trie import Trie


def modify_items(text, eos):

    return f'{text}\n{eos}'


def load_titles_to_trie_from_json(input_file, tokenizer, frequency_scale):
    trie = Trie(tokenizer=tokenizer, frequency_scale=frequency_scale)
    with open(input_file, 'r', encoding='utf-8') as file:
        item_frequency_dict = json.load(file)


    for title_name, frequency in tqdm(item_frequency_dict.items(), desc="Loading titles to Trie", mininterval=10.0):

        modify_title = modify_items(title_name, tokenizer.eos_token)
        trie.insert(modify_title, frequency)

    trie.frequency_scale = trie.total_frequency
    return trie


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath=args.ckpt_dir,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        every_n_epochs=1
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval='step'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)


    model = MInterface(**vars(args))

    if args.mode == 'train' and args.use_igd:
        print("Building Trie for IGD-Tuning...")
        if not args.item_freq_path or not os.path.exists(args.item_freq_path):
            raise ValueError("Please provide a valid --item_freq_path (JSON file with POI frequencies) for IGD-Tuning")


        rf_item_trie = load_titles_to_trie_from_json(
            args.item_freq_path,
            tokenizer=model.llama_tokenizer,
            frequency_scale=1.0
        )
        rf_item_trie.compute_information_gain()


        model.rf_item_trie = rf_item_trie
        print("Trie built and attached to model.")

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(args.ckpt_path))

    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))

    args.max_steps = len(data_module.trainset) * args.max_epochs // (args.accumulate_grad_batches * args.batch_size)

    logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    args.logger = logger
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    trainer = Trainer.from_argparse_args(args)

    if args.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=data_module, min_lr=1e-10, max_lr=1e-3,
                                          num_training=100)
        fig = lr_finder.plot(suggest=True)
        fig_path = "lr_finder.png"
        fig.savefig(fig_path)
        print("Saving to {}".format(fig_path))
        model.hparams.lr = lr_finder.suggestion()

    if args.mode == 'train':
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=-1, type=int)
    parser.add_argument('--precision', default='bf16', type=str)
    parser.add_argument('--amp_backend', default="native", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)
    parser.add_argument('--dataset', default='nyc', type=str)
    parser.add_argument('--data_dir', default='data/ref/ca', type=str)
    parser.add_argument('--loss', default='lm', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--log_dir', default='movielens_logs', type=str)
    parser.add_argument('--padding_item_id', default=1682, type=int)
    parser.add_argument('--llm_path', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    parser.add_argument('--llm_tuning', default='lora', choices=['lora', 'freeze', 'freeze_lora'], type=str)
    parser.add_argument('--peft_dir', default=None, type=str)
    parser.add_argument('--peft_config', default=None, type=str)
    parser.add_argument('--lora_r', default=8, type=float)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--use_igd', action='store_true', help="Use IGD-Tuning")
    parser.add_argument('--beta', default=0.1, type=float, help="Beta for IGD-Tuning (weight for zero-IG tokens)")
    parser.add_argument('--item_freq_path', type=str, default=None,
                        help="Path to json file containing item frequencies")

    args = parser.parse_args()

    main(args)