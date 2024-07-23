export CUDA_VISIBLE_DEVICES=0

MODEL=lima_noins_partial_noins7B
MODEL=$1

# Big-Bench Hard
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/${MODEL}/ \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# CODEX
# Evaluating noins 7B model using temperature 0.8 to get the pass@10 score without chat format
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/${MODEL}_temp_0_8_nochat \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --use_vllm

# GSM
# Evaluating LIMA noins 7b model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/${MODEL}-cot-8shot \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# IF EVAL
# Evaluating noins 7B model using chat format
python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/${MODEL} \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# MMLU
# Evaluating LIMA noins 7B model using 0 shot and chat format
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/${MODEL}-cot-0shot \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# TruthfulQA
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/${MODEL}/ \
    --model_name_or_path  ${MODEL} \
    --tokenizer_name_or_path  ${MODEL} \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
