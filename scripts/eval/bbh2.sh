# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# evaluating baseline 7B model using chain-of-thought and chat format
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/lima_baseline_7B/ \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# evaluating noins 7B model using chain-of-thought and chat format
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/lima_noins_7B/ \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


