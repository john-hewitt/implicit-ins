# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
 export CUDA_VISIBLE_DEVICES=0

# Evaluating baseline 7B model using chat format
python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/lima_baseline_7B \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# Evaluating noins 7B model using chat format
python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/lima_noins_7B \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

