export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=1B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.


# Sphinx1 sigh
export HF_DATASETS_CACHE='/scr/johnhew/dataset_cache'
export STANZA_RESOURCES_DIR='/u/scr/johnhew/stanza_resources'
export HF_HOME='/scr/johnhew/hf_home/'
export TRANSFORMERS_CACHE='/scr/johnhew/hf_cache'
export XDG_CACHE_HOME=/scr/johnhew/

source ~/switch-cuda/switch-cuda.sh 11.7

# 1 epoch since dataset already repeated 10x
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --use_flash_attn \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --use_slow_tokenizer \
    --train_file data/processed/lima/lima_no_instruction_plus_refusal.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 15 \
    --output_dir output/lima_noins_refusal_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
