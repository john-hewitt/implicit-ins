## Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0
#
## Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score with chat format via HumanEvalPack
## This leads to ~same scores as without chat format, see https://github.com/allenai/open-instruct/pull/99#issuecomment-1953200975
#python -m eval.codex_humaneval.run_eval \
#    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#    --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
#    --use_chat_format \
#    --eval_pass_at_ks 1 5 10 20 \
#    --unbiased_sampling_size_n 20 \
#    --temperature 0.1 \
#    --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
#    --model ../checkpoints/tulu_7B/ \
#    --tokenizer ../checkpoints/tulu_7B/ \
#    --use_vllm
#
## Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score without chat format
#python -m eval.codex_humaneval.run_eval \
#    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#    --eval_pass_at_ks 1 5 10 20 \
#    --unbiased_sampling_size_n 20 \
#    --temperature 0.1 \
#    --save_dir results/codex_humaneval/tulu_7B_temp_0_1_nochat \
#    --model ../checkpoints/tulu_7B/ \
#    --tokenizer ../checkpoints/tulu_7B/ \
#    --use_vllm


# Evaluating baseline 7B model using temperature 0.8 to get the pass@10 score without chat format
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/lima_baseline_7B_temp_0_8_nochat \
    --model output//lima_baseline_7B/ \
    --tokenizer output//lima_baseline_7B/ \
    --use_vllm

# Evaluating noins 7B model using temperature 0.8 to get the pass@10 score without chat format
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/lima_noins_7B_temp_0_8_nochat \
    --model output//lima_noins_7B/ \
    --tokenizer output//lima_noins_7B/ \
    --use_vllm


