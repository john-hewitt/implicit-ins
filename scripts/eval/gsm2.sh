# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

# Evaluating LIMA baseline 7b model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/lima_baseline_7B-cot-8shot \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# Evaluating LIMA noins 7b model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/lima_noins_7B-cot-8shot \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm


## Evaluating llama2 chat model using chain-of-thought and chat format
#python -m eval.gsm.run_eval \
#    --data_dir data/eval/gsm/ \
#    --max_num_examples 200 \
#    --save_dir results/gsm/llama2-chat-7B-cot-8shot \
#    --model ../hf_llama2_models/7B-chat \
#    --tokenizer ../hf_llama2_models/7B-chat \
#    --n_shot 8 \
#    --use_chat_format \
#    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#    --use_vllm
