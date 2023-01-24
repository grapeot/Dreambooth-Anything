export LR=5e-6

accelerate launch \
--mixed_precision=bf16 \
--num_processes=1 \
--num_machines=1 \
--dynamo_backend=no \
scripts/train_dreambooth_depth.py \
--mixed_precision=bf16 \
--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-depth  \
--pretrained_txt2img_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
--instance_data_dir=data/Lycoris \
--instance_prompt="an anime in LYCORISANIME style" \
--class_data_dir=data/LycorisClass \
--class_prompt='an anime' \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--output_dir=data/models/sd21_dpeth_Lycoris500_class_scheduler_${LR} \
--resolution=512 \
--instance_prompt_shuffle_prob=0.0 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--learning_rate=$LR \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_class_images=400 \
--max_train_steps=25000 \
--checkpointing_steps=5000 \
--drop_incomplete_batches \
--use_8bit_adam \
--num_workers=1 \
--pin_memory \
--persistant_workers \
--prefetch_factor 2 \
--save_sample_prompt='an anime in LYCORISANIME style' \
--save_input_folder=data/NichijouVideo/frames
# --resume_from_checkpoint=latest
