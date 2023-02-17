export LR=2e-8

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
--output_dir="data/models/sd21_dpeth_Lycoris6k_${LR}_scheduler" \
--resolution=512 \
--instance_prompt_shuffle_prob=0.0 \
--train_batch_size=32 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--scale_lr \
--learning_rate=$LR \
--lr_scheduler="linear" \
--lr_warmup_steps=150 \
--num_class_images=0 \
--max_train_steps=90000 \
--checkpointing_steps=5000 \
--drop_incomplete_batches \
--use_8bit_adam \
--num_workers=1 \
--pin_memory \
--persistant_workers \
--prefetch_factor 2 \
--save_sample_prompt='an anime in LYCORISANIME style' \
--save_input_folder=data/NichijouVideo/sampleFrames \
--save_infer_steps=400 \
--class_data_dir=data/LycorisClass \
# --resume_from_checkpoint=latest
# --train_text_encoder \