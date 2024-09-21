#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
MODELNAME=baseline
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.

DSNAME=recipe3e-6
epochs=20
seed=5
model=olmo${DSNAME}${MODELNAME}${MODEL_SIZE}ep${epochs}_seed${seed}

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    --main_process_port 29524 \
    open_instruct/finetune.py \
    --model_name_or_path allenai/OLMo-7B-hf \
    --use_flash_attn \
    --tokenizer_name allenai/OLMo-7B-hf \
    --use_slow_tokenizer \
    --train_file data/processed/kaggle_food_recipes/kfr.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 3e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs ${epochs} \
    --output_dir output/${model}/ \
    --with_tracking \
    --report_to tensorboard \
    --seed ${seed} \
    --logging_steps 1


export CUDA_VISIBLE_DEVICES=0

python -m eval.val_eval.run_eval --model_name_or_path output/${model}/  --tokenizer_name_or_path output/${model}/ --save_dir results/val_eval/${model}/      --eval_batch_size 10          --use_chat_format     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format --use_vllm

alpaca_eval --model_outputs results/val_eval/${model}/${model}-greedy-long-output.json --reference_outputs eval/val_eval/val-gpt3.5-2.json

# Test alpaca eval
python -m eval.alpaca_farm.run_eval --model_name_or_path output/${model}/  --tokenizer_name_or_path output/${model}/ --save_dir results/alpaca_farm/${model}/      --eval_batch_size 10          --use_chat_format     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format --use_vllm
baseline_model=olmolima3e-6baseline${MODEL_SIZE}ep15_seed${seed}
alpaca_eval --model_outputs results/alpaca_farm/${model}/${model}-greedy-long-output.json --reference_outputs results/alpaca_farm/${baseline_model}/${baseline_model}-greedy-long-output.json
