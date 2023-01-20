accelerate launch \
--mixed_precision=bf16 \
--num_processes=1 \
--num_machines=1 \
--dynamo_backend=no \
scripts/train_dreambooth_depth.py \
--mixed_precision=bf16 \
--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth  \
--pretrained_txt2img_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--instance_data_dir=Lycoris \
--instance_prompt="an anime in LYCORISANIME style" \
--class_data_dir=LycorisClass \
--output_dir=sd21_dpeth_Lycoris500 \
--resolution=512 \
--instance_prompt_shuffle_prob=0.0 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=2e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_class_images=0 \
--max_train_steps=50000 \
--checkpointing_steps=10000 \
--drop_incomplete_batches \
--use_8bit_adam \
--num_workers=1 \
--pin_memory \
--persistant_workers \
--prefetch_factor 2
#--train_text_encoder \
