# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=False

# Evaluating LIMA baseline 7b model using and chat format
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path output//lima_baseline_7B20e/ \
    --tokenizer_name_or_path output//lima_baseline_7B20e/ \
    --save_dir results/alpaca_farm/lima_baseline_7B20e/ \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating LIMA noins 7b model using and chat format
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path output//lima_noins_7B20e/ \
    --tokenizer_name_or_path output//lima_noins_7B20e/ \
    --save_dir results/alpaca_farm/lima_noins_7B20e/ \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
