import re

import torch

import pytorch_lightning as pl

from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from pandas.core.frame import DataFrame
import os.path as op
import os

from optims import LinearWarmupCosineLRScheduler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


class MInterface(pl.LightningModule):
    def __init__(self,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)

    def forward(self, batch, input_embeds):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0), -100)  # transformers=4.34.0
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch, temperature=0.8, do_sample=False, num_beams=1, max_gen_length=7, min_gen_length=1,
                 repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
        )
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        # replace_llama_attn(use_flash_attn=True, use_full=False)
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        # if batch["flag"]:
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = False
        # else:
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = True
        input_embeds = self.wrap_emb(batch)
        out = self(batch, input_embeds)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_content = {
            "generate": [],
            "real": [],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # replace_llama_attn(inference=True)
        generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch['answer'][i].split("\n")[0]
            generate = generate.strip().split("\n")[0]
            #print(generate)
            output.append((generate, real))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)

    def on_validation_epoch_end(self):
        df = DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(self.val_content)
        metric = hr
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content = {
            "generate": [],
            "real": [],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # replace_llama_attn(inference=True)
        generate_output = self.generate(batch)
        output = []
        #real = batch['answer'][0].split("\n")[0]
        #generate=generate_output[0].strip().split("\n")[0]
        #output.append((generate, real))
        for i, generate in enumerate(generate_output):
            real = batch['answer'][i].split("\n")[0]
            # print(generate)
            generate=generate.strip().split("\n")[0]
            output.append((generate, real))
        return output

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)

    def on_test_epoch_end(self):
        df = DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(self.test_content)
        metric = hr
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                               max_step=max_step,
                                                               min_lr=self.hparams.lr_decay_min_lr,
                                                               init_lr=self.hparams.lr,
                                                               warmup_steps=warmup_steps,
                                                               warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass

    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            use_flash_attention_2="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj'])  # 这一部分的设定和LLM4POI有出入
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        print('Loading LLAMA Done')

    def wrap_emb(self, batch):
        input_ids = batch["tokens"].input_ids
        input_embeds = self.llama_model.get_input_embeddings()(input_ids)
        poi_token_id = self.llama_tokenizer("[PH]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        poi_embeds = batch["poi_embeds"]
        if not batch["flag"]: #修改过
            for i in range(input_ids.shape[0]):
                if (input_ids[i] == poi_token_id).nonzero().shape[0] > 0:
                    idx_tensor = (input_ids[i] == poi_token_id).nonzero().view(-1)
                    for idx, poi_emb in zip(idx_tensor, poi_embeds[i]):
                        idx = idx.item()
                        input_embeds[:, idx, :] = poi_emb.unsqueeze(0)
        return input_embeds

    def calculate_hr1(self, eval_content):
        correct_num = 0
        valid_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            #real = re.sub(r'[^0-9]', '', real)
            #print(real)
            total_num += 1
            generate = re.sub(r'[^0-9]', '', generate)
            #print(generate)
            if generate.isdigit() and 0 <= int(generate) <= 7833: # nyc 4980 tky 7833 ca 9690
                valid_num += 1
            if generate == real:
                correct_num += 1
        valid_ratio = valid_num / total_num
        if valid_num > 0:
            hr1 = correct_num / total_num
        else:
            hr1 = 0
        return valid_ratio, hr1
