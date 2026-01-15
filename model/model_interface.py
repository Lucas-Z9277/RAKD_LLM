import math
import re
import torch
import torch.nn as nn  # 新增
import pytorch_lightning as pl
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        self.rf_item_trie = None
        self.igd_beta = kargs.get('beta', 1.0)
        self.use_igd = kargs.get('use_igd', False)

    def forward(self, batch, input_embeds):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100) # transformers=4.28.0
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0), -100)  # transformers=4.34.0
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch, temperature=0.8, do_sample=False, num_beams=5, max_gen_length=7, min_gen_length=1,
                 repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=5):
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
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)

        input_embeds = self.wrap_emb(batch)

        if self.use_igd and self.rf_item_trie is not None:
            out = self(batch, input_embeds)
            logits = out.logits
            targets = batch["tokens"].input_ids.masked_fill(
                batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            targets = targets.masked_fill((batch["tokens"].token_type_ids == 0), -100)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            non_pad_mask = shift_labels != -100
            ig_weights = torch.ones_like(per_token_loss, dtype=torch.float32)

            all_ig_tensors = []
            for i, seq_labels in enumerate(shift_labels):
                valid_labels = seq_labels[non_pad_mask[i]].tolist()
                if valid_labels:
                    ig_list = self.rf_item_trie.get_sequence_ig(valid_labels)
                    if ig_list and ig_list[-1] == float('-inf'):
                        ig_list = [0] * len(valid_labels)

                    if len(ig_list) != len(valid_labels):
                        ig_list = [0] * len(valid_labels)

                    all_ig_tensors.append(torch.tensor(ig_list, dtype=torch.float32, device=per_token_loss.device))
                else:
                    all_ig_tensors.append(torch.tensor([], dtype=torch.float32, device=per_token_loss.device))

            if all_ig_tensors:
                pass

            start_idx = 0
            for i, seq_labels in enumerate(shift_labels):
                mask = non_pad_mask[i]
                valid_len = mask.sum().item()
                if valid_len > 0:
                    current_ig = all_ig_tensors[i]
                    sample_weights = torch.ones(valid_len, device=per_token_loss.device)
                    zero_mask = current_ig <= 0  # IG <= 0 的 token
                    pos_mask = current_ig > 0  # IG > 0 的 token

                    sample_weights[zero_mask] = self.igd_beta
                    sample_weights[pos_mask] = 1.0
                    ig_weights[i, mask] = sample_weights

            new_weights = ig_weights[non_pad_mask]
            loss_sum = (per_token_loss[non_pad_mask] * new_weights).sum()
            weight_sum = new_weights.sum() + 1e-8
            loss = loss_sum / weight_sum

        else:
            out = self(batch, input_embeds)
            loss = self.configure_loss(out)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss
