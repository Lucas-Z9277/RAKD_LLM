nohup python main.py \
--mode train \
--batch_size 1 \
--accumulate_grad_batches 4 \
--llm_tuning lora \
--llm_path Llama-2-7b-hf \
--dataset nyc \
--ckpt_dir ./checkpoints/nyc/short_rest \
--output_dir ./output/nyc/ \
--log_dir nyc_logs \
--lr_warmup_start_lr 2e-6 \
--lr 2e-5 \
--lr_decay_min_lr 2e-6 \
--max_epochs 4 \
--use_igd \
--beta 0.1 \
--item_freq_path ./item_freq.json \
> train_nyc_short_rest.log 2>&1 &