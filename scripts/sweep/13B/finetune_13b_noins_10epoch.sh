#! /bin/bash

__conda_setup="$('/u/scr/johnhew/miniconda3/' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/scr/johnhew/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/scr/johnhew/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/u/scr/johnhew/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate poi6
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_SIZE=13B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

MODELNAME=noins

## Lora training
#accelerate launch \
#    --num_machines 1 \
#    --num_processes $NUM_GPUS \
#    --main_process_port 29505 \
#    open_instruct/finetune.py \
#    --model_name_or_path meta-llama/Llama-2-13b-hf \
#    --tokenizer_name meta-llama/Llama-2-13b-hf \
#    --gradient_checkpointing \
#    --use_8bit_optimizer \
#    --use_qlora \
#    --use_lora \
#    --use_flash_attn \
#    --lora_rank 64 \
#    --lora_alpha 16 \
#    --lora_dropout 0.1 \
#    --use_slow_tokenizer \
#    --train_file data/processed/lima/lima_no_instruction.jsonl \
#    --max_seq_length 2048 \
#    --preprocessing_num_workers 128 \
#    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#    --learning_rate 1.5e-4 \
#    --lr_scheduler_type linear \
#    --warmup_ratio 0.03 \
#    --weight_decay 0. \
#    --num_train_epochs 10 \
#    --output_dir output/lima_${MODELNAME}_${MODEL_SIZE}_qlora_10epoch/ \
#    --with_tracking \
#    --report_to tensorboard \
#    --logging_steps 1 &&
#
#python open_instruct/merge_lora.py \
#    --base_model_name_or_path meta-llama/Llama-2-13b-hf \
#    --lora_model_name_or_path output/lima_${MODELNAME}_${MODEL_SIZE}_qlora_10epoch/ \
#    --output_dir output/lima_${MODELNAME}_${MODEL_SIZE}_qlora_merged_10epoch/ \
#    --qlora \
#    --save_tokenizer
#
export CUDA_VISIBLE_DEVICES=0

model=lima_${MODELNAME}_${MODEL_SIZE}_qlora_merged_10epoch; python -m eval.val_eval.run_eval --model_name_or_path output/${model}/  --tokenizer_name_or_path output/${model}/ --save_dir results/val_eval/${model}/      --eval_batch_size 10          --use_chat_format     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format --use_vllm

alpaca_eval --model_outputs results/val_eval/${model}/${model}-greedy-long-output.json --reference_outputs eval/val_eval/val-gpt3.5-2.json
