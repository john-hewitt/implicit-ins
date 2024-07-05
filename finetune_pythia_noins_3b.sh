export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=3B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    --main_process_port 29503 \
    open_instruct/finetune.py \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --use_flash_attn \
    --tokenizer_name  EleutherAI/pythia-2.8b \
    --train_file data/processed/lima/lima_no_instruction.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 20 \
    --output_dir output/lima_noins_${MODEL_SIZE}pythia/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
