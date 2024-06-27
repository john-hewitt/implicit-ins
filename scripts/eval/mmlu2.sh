# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating LIMA baseline 7B model using 0 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/lima_baseline_7B-cot-0shot \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating LIMA baseline 7B model using 5 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/lima_baseline_7B-cot-5shot \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format



# Evaluating LIMA noins 7B model using 0 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/lima_noins_7B-cot-0shot \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating LIMA noins 7B model using 5 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/lima_noins_7B-cot-5shot \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
