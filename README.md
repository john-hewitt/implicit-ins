
# Instruction Following without Instruction Tuning

This codebase contains the code for the paper _[Instruction Following without Instruction Tuning](https://arxiv.org/pdf/2409.14254)_, by John Hewitt, Nelson F. Liu, Christopher D. Manning, and Percy Liang.

This codebase is a fork of the [open-instruct](https://github.com/allenai/open-instruct) repository, modified to implement experiments in the paper _Instruction Following without Instruction Tuning._
As such, it can be used for instruction tuning, though likely unless you're interested in the details of our experiments, you're better off using the original open-instruct repository.

The core result of this work is that some adaptations (finetuning) to language models that don't _seem_ like instruction tuning still _implicitly_ instruction-tune the model, in the sense that the base model didn't follow instructions, and the adapted model roughly does.
Training on responses only (without any instructions) or training on a single-task distribution like poetry generation, both yield instruction following.
We even hand-write a 3-rule rule-based model that, in a product with a pretrained distribution, causes instruction following.

These results were quite curious to us! You can use the scripts below to replicate our experiments.
Then below our notes is the original README for the open-instruction repository when we forked it, for reference.

<p align="center" width="100%">
      <img src="images/fig1.png" alt="Training on responses only, or single-task distributions, or even using a rule-based adapter, all cause pretrained models to follow instructions." style="min-width: 200px; display: block; margin: auto;">
</p>

## Downloading and processing data.
We use various transformed versions of datasets that have access restrictions (LIMA in particular) so there's a bit of a process to get the data.

First, run `scripts/prepare_train_data.sh`, which will download the LIMA and Stanford alpaca datasets. (For LIMA you'll need your `HF_TOKEN` set and approval on the dataset.)

Now create the no-instructions LIMA:

    cd data/processed/lima
    python remove_instructions.py

Now create the single-task finetuning datasets

    # Grade School Math
    cd ../gsm_train
    python make_gsm.py

    # Mostly Basic Python Programming
    cd ../mbpp
    python make_mbpp.py

    # Recipes (https://huggingface.co/datasets/Hieu-Pham/kaggle_food_recipes)
    cd ../kaggle_food_recipes
    python make_kfr.py

    # Chess (https://huggingface.co/datasets/patrickfrank1/chess-pgn-games)
    cd ../pgn
    python make_pgn.py

    # Poetry (https://huggingface.co/datasets/merve/poetry)
    cd ../poetry
    python make_poetry.py

## Getting Started

Make your preferred environment (I use miniconda) and then install some required packages:

     pip install -r requirements.txt 

Instruction-tune a model, and then evaluate on our dev set, to see if everything's working:

     bash scripts/iclr2025/expt1_llama/ins/finetune_5e.sh

## Response Tuning

Make sure to set your `OPENAI_API_KEY`, since the following scripts will automatically run
AlpacaEval using GPT-4 as a judge.

First, run the validation experiments to see which number of epochs of training we'll pick.
These scripts train Llama-2 models on LIMA and then validate using our internal val set and
AlpacaEval against GPT-3.5 outputs.

    # Instruction-tuned models
    for s in scripts/iclr2025/expt1_llama/ins/finetune*e.sh; do bash $s; done
    # Response-tuned models
    for s in scripts/iclr2025/expt1_llama/res/finetune*e.sh; do bash $s; done

The results are stored here:

    for ep in 5 7 10 15 20; do
      cat results/val_eval/limabaseline7Bep${ep}/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
    done

The `finetune_seed*.sh` files already have the epochs set to what worked best for our validation run,
or you can update them based on your validation run. First run the instruction-tuning test runs:

    for s in scripts/iclr2025/expt1_llama/ins/finetune_seed*; do bash $s; done

It's important that these all complete first before running the response-tuning experiments,
since the response-tuning experiments will run alpaca-eval head-to-head against the outputs
from the corresponding seed of the instruction-tuning experiments.
So, `finetune_seed1.sh` of the response-tuning experiments will compare against the outputs of
seed 1 of instruction-tuning. You also have to edit the `res/finetune_seed*.sh` test results
to point to the outputs of the test models if you changed the seed from what we set.

Now run the response-tuning and evaluations:

    for s in scripts/iclr2025/expt1_llama/res/finetune_seed*; do bash $s; done

Now the results that we put in Table 1 (Llama-2-7B Win Rate vs. Instruction Tuning, 43.2%)
should appear (up to randomness; these results aren't going to be identically replicable
but should be replicable in distribution) at:

    cat results/alpaca_farm/limaresponse7Bep5_seed*/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv

## Single-Task Finetuning
The process for running the single-task finetuning experiments is largely the same as for the response-tuning
experiments, just with different paths.

First we run the valiation experiments:
We use `gsm` here as an example (training on 1k of the Grade School Math 8k) experiments, but other options are `mbpp, pgn, poetry, recipe`. 

    bash scripts/iclr2025/expt2_llama/gsm/finetune*e.sh

    for ep in 5 7 10 15 20; do
      cat results/val_eval/gsmbaseline7Bep${ep}/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
    done

And then run the test seeds (again after changing the epoch counts in the test scripts depending on your validation results):

    for s in scripts/iclr2025/expt2_llama/gsm/finetune_seed*; do bash $s; done

    cat results/alpaca_farm/gsmbaseline7Bep7_seed*/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv

Note that these test scripts must be run _after_ you ran the instruction-tuning test scripts from the response tuning
section, since the evaluation scripts will compare the output of the GSM-trained models against the outputs of
the instruction-tuned models.

## Rule-Based Model

To run our rule-base models, we'll call our generation script (which generates outputs from a model for a given set of instructions) with special flags that add the rules.

    options=eos-uniform-diversity
    model=meta-llama/Llama-2-7B-hf
    python -m eval.alpaca_farm.run_eval \
      --model_name_or_path ${model} \
      --tokenizer_name_or_path ${model} \
      --save_dir results/alpaca_farm/`basename ${model}`-rules-${options} \
      --eval_batch_size 1 \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
      --add_rule_based_helper ${options}

To run the ablations without some of the rules, set the options as, e.g., `eos-uniform` to remove the diversity rule.

To evaluate these outputs, run, e.g.,

    for s in 1 2 3 4 5; do
      alpaca_eval \
        --model_outputs results/alpaca_farm/Llama-2-7B-hf-rules-eos-uniform-diversity/Llama-2-7B-hf-greedy-long-output.json\
        --reference_outputs results/alpaca_farm/limabaseline7Bep15_seed${s}/limabaseline7Bep15_seed${s}-greedy-long-output.json\
        --output_path results/alpaca_farm/Llama-2-7B-hf-rules-eos-uniform-diversity/result_seed${s}
    ; done

This runs `alpaca_eval` against seed `s` of the instruction-tuned model.

For our non-EOS models, we constrain to at most 512 new tokens with the flag `--max_new_tokens 512` (otherwise it takes forever since they largely don't stop generating.) This also improves their win rates.

## Response Ranking Experiment (Section 4.2)

Score real responses and random responses for each of three models:

    for model in output/limabaseline7Bep15_seed1/ output/olmolima3e-6baseline7Bep15_seed1/ meta-llama/Llama-2-7B-hf allenai/OLMo-7B-hf; do  python open_instruct/ratio_eval.py --model_name_or_path ${model} --tokenizer_name output/limabaseline7Bep15_seed1/  --train_file data/processed/stanford_alpaca/stanford_alpaca_data.jsonl --max_seq_length 1024  --per_device_train_batch_size 1 --max_examples 1000 --output_path `echo $model | sed 's|/|-|g'`.jsonl; done

Print response ranking percents:

    python plot_ratios.py output-limabaseline7Bep15_seed1- 

## Visualization of response similarity (Section 5.2)

Assuming your test results are at the same place as indicated via the config files in this repository. (if not, edit the python file):

    python open_instruct/plot_embeds.py gsm

## Citation

If you found this useful, please cite the papers for the original repository:

```bibtex
@misc{wang2023far,
   title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources}, 
   author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
   year={2023},
   eprint={2306.04751},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```

```bibtex
@misc{ivison2023camels,
      title={Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2}, 
      author={Hamish Ivison and Yizhong Wang and Valentina Pyatkin and Nathan Lambert and Matthew Peters and Pradeep Dasigi and Joel Jang and David Wadden and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
      year={2023},
      eprint={2311.10702},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

and feel free to cite our paper as well:

@misc{hewitt2024instruction,
   title={Instruction Following without Instruction Tuning}, 
   author={John Hewitt and Nelson F. Liu and Christopher D. Manning and Percy Liang},
   year={2024},
   eprint={pdf/2409.14254},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```


# (Forked From) Training Open Instruction-Following Language Models
Below I leave the README from the forked repository, unchanged.

This repo serves as an open effort on instruction-tuning popular pretrained language models on publicly available datasets. We release this repo and will keep updating it with:

1. Code for finetuning language models with latest techniques and instruction datasets in a unified format.
2. Code for running standard evaluation on a range of benchmarks, targeting for differnt capabilities of these language models.
3. Checkpoints or other useful artifacts that we build in our exploration.

Please see our first paper [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) for more thoughts behind this project and our initial findings. Please see our second paper [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702) for newer results using Llama-2 models and direct preference optimization. We are still working on more models, so stay tuned for future work!

<p align="center" width="100%">
      <img src="images/tulu_logo.png" alt="Tülu (a hybrid camel) represents a suite of LLaMa models that we built by fully-finetuning them on a strong mix of datasets." style="width: 20%; min-width: 200px; display: block; margin: auto;">
</p>

## News

- [2023-11-27] We released [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702). Check out our models [here](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101). We have added a DPO finetuning script for replicating our results.
- [2023-09-26] We switched to use the official [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) library to run AlpacaFarm evaluation but use regenerated longer reference outputs. This will change our numbers reported in the paper. We will update the paper soon.
- [2023-09-25] Supported using [vLLM](https://github.com/vllm-project/vllm/) for our evaluations, which speeds up the evaluation by 10x.
- [2023-09-17] Supported [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) finetuning. See [here](#parameter-efficient-finetuning) for more details.
- [2023-08-18] Added support for [ToxiGen](https://github.com/microsoft/TOXIGEN)/[TrutufulQA](https://github.com/sylinrl/TruthfulQA) evaluation. Check our `scripts/eval/` for examples of running them.
- [2023-08-08] Supported several new instruction dataset, including [LIMA](https://huggingface.co/datasets/GAIR/lima) / [WizardLM](https://github.com/nlpxucan/WizardLM) / [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca). See the [preparation script](./scripts/prepare_train_data.sh) for details. Performance hasn't been evaluated yet.
- [2023-08-06] Supported LLaMa 2 finetuning and FlashAttention-2 by bumping the version of transformers and many other dependencies.
- [2023-06-29] Added [licensing info](#licensing) for our released models.
- [2023-06-09] Released Tülu (a suite of LLaMa models fully-finetuned on a strong mix of datasets) and many other checkpoints on HuggingFace [[Links]](#released-checkpoints).
- [2023-06-09] Initial release of the codebase containing the training and evaluation code for our [arxiv paper](https://arxiv.org/abs/2306.04751).

## Setup

To run training, evaluation, or inference for our finetuned models, you need to install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```

If you just want the dependencies for the weight diff script, use:
```bash
pip install -r weight-diff-requirements.txt
```

If you'd like to experiment with AI2's [OLMo](https://huggingface.co/allenai/OLMo-7B) models, you should also install:

```bash
pip install ai2-olmo
```

If you'd like to run experiments within a Docker environment, you can create one using:

```bash
docker build --build-arg CUDA=11.8.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t <your tag here>
```

If you are internally at AI2, you can use this pre-built beaker image [here](https://beaker.org/im/01HQ1PMA9YCVKXYN6BHP5EYV5E/details).

## Training

### Dataset preparation

We include a collection of representative instruction datasets in our exploration and are adding new ones to our list. We unify them into the same chatting format. To download and prepare these datasets, simply run the following command:

```bash
./scripts/prepare_train_data.sh
```

Please check these datasets for licenses and restrictions around their use!

You can also find the processed [Tulu v1](https://huggingface.co/datasets/allenai/tulu-v1-sft-mixture) and [Tulu v2](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) SFT datasets on HuggingFace.

### Model preparation

Generally, most huggingface-compatible causal language models should work fine with our codebase, potentially with some adjusting for different tokenizers etc. Some models may require addtional requests to download. E.g., for LLaMa 1 and 2, please consult [the Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/llama) for requesting access and converting them to a huggingface-compatible format.

### Finetuning

You can use the following command to run instruction tuning (finetuning a pretrained model to follow instructions):

```bash
./scripts/finetune_with_accelerate.sh
```

Make sure to adjust `model_name_or_path`, `tokenizer_name`, `train_file`, and `output_dir` to your models / data / setting. By default, this uses `deepspeed` with `accelerate`.

### Parameter-Efficient Finetuning

We support [LoRA](https://arxiv.org/abs/2106.09685) finetuning, wherein only a small number of parameters are updated, resulting in faster and cheaper training. For even more efficiency, we also support [QLoRA](https://arxiv.org/abs/2305.14314) finetuning, wherein the non-trained (underlying) model parameters are quantised during 4-bit training. This means you can train a 70b Llama model on a single 80GB A100! Please refer to the respective papers for more details.

Please also note you cannot currently run QLoRA with model parallelism - only data-parallel training is supported, so you cannot train a model that does not fit on one GPU. For LoRA, you can use deepspeed + zero-3 to achieve model parallelism (and FSDP is not currently supported).

Please see `./scripts/finetune_lora_with_accelerate.sh` and `./scripts/finetune_qlora_with_accelerate.sh` for example hyperparameters. We found a larger rank (e.g. 256) and higher learning rate (e.g. 2e-4) worked best. Additionally, we found that QLoRA tended to always achieve similar results to LoRA, while LoRA itself sometimes fell behind full-finetuning, especially in long, complex generation tasks. However, for most purposes, LoRA training essentially matches full-finetuning performance. We recommend merging modules learnt with QLoRA into a dequantised model (run our merge script with the `--qlora` flag).

## DPO Finetuning

For an example of how to fully finetune a model with DPO, see `scripts/dpo_train_with_accelerate.sh`. Note you will require at least 8 80GB A100s to be able to train a 7b size model, and will require more compute for anything larger. We have not tested multi-node training with this script, but it should work.

Our script also supports PEFT training with QLoRA. See `scripts/dpo_train_with_qlora.sh` for an example. We have not trained models with this, so it may require additional hyperparameter tuning to achieve reasonable results.

## Released Checkpoints

Our checkpoints can be found:

- [Here](https://huggingface.co/collections/hamishivi/tulu-v1-suite-655138c3743e6349aaa07d7d) for all Tulu v1 models.
- [Here](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) for all Tulu v2 models.


### Weight diff script

Our Tulu V1 models were released as weight diffs (due to LLaMa 1 license). We use a slightly modified form of the [Alpaca weight diff script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py), which runs the same.

To merge a model:
1. Download the relevant LLaMa model and convert it to Hugging Face format (see above).
2. Download our repository and install the right dependencies (see above).
3. Download the model diff you want.
4. Run the command below:

```bash
python scripts/weight_diff.py recover --path_raw ${hf_llama_path} --path_tuned ${output_path} --path_diff ${diff_location}
```

## Evaluation

### Benchmark-based eval

We provide the scripts for running evaluation of Huggingface/OpenAI models on a list of standard benchmarks targeting for the core capabilities of large language models. These benchmakrs include:

- [MMLU](https://github.com/hendrycks/test)
- [Grade School Math (GSM)](https://github.com/openai/grade-school-math)
- [Big-Bench Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main)
- [TydiQA](https://github.com/google-research-datasets/tydiqa)
- [Codex HumanEval](https://github.com/openai/human-eval/tree/master)
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [ToxiGen](https://github.com/microsoft/TOXIGEN)
- [XSTest](https://github.com/paul-rottger/exaggerated-safety/)
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)

We are working on including more promising benchmarks into this list. Please stay tuned!

You can use the following script to download all the evaluation data:

```bash
./scripts/prepare_eval_data.sh
```

Evaluation scripts for different datasets are put under `./scripts`. For example, you can use the following command to run the MMLU evaluation script:

```bash
./scripts/eval/mmlu.sh
```

### Human evaluation

We release our human evaluation interface and collected annotations in the `./human_eval` folder. Please see the corresponding [README](./human_eval/README.md) for more details.

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The license we use for V1 models released (along with the base model licenses) can be found in [model_licenses/tulu_license.txt](./model_licenses/tulu_license.txt) - just replace `<MODELNAME>` with the actual model name (i.e., the name on HuggingFace).

V2 models are licensed under the [low-risk AI2 ImpACT license](https://allenai.org/licenses/impact-lr). See [here](https://allenai.org/impact-license) for more details.

## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{wang2023far,
   title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources}, 
   author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
   year={2023},
   eprint={2306.04751},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```

```bibtex
@misc{ivison2023camels,
      title={Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2}, 
      author={Hamish Ivison and Yizhong Wang and Valentina Pyatkin and Nathan Lambert and Matthew Peters and Pradeep Dasigi and Joel Jang and David Wadden and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
      year={2023},
      eprint={2311.10702},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
