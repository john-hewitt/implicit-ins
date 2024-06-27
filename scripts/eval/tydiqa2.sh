# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0



# Evaluating baseline 7B model, with gold passage provided
# For baseline, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/lima_baseline_7B_gold \
    --model output//lima_baseline_7B \
    --tokenizer  output//lima_baseline_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# Evaluating baseline 7B model, with no context provided (closed-book QA)
# For baseline, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/lima_baseline_7B_closed \
    --model output//lima_baseline_7B \
    --tokenizer  output//lima_baseline_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --no_context \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format



# Evaluating noins 7B model, with gold passage provided
# For noins, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/lima_noins_7B_gold \
    --model output//lima_noins_7B \
    --tokenizer  output//lima_noins_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# Evaluating noins 7B model, with no context provided (closed-book QA)
# For noins, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/lima_noins_7B_closed \
    --model output//lima_noins_7B \
    --tokenizer  output//lima_noins_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --no_context \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

