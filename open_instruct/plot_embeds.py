# Drawn partially from https://huggingface.co/nomic-ai/nomic-embed-text-v1

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json

argp = argparse.ArgumentParser()
#argp.add_argument('sft_datsaet')
#argp.add_argument('model_alpaca_outputs')
#argp.add_argument('ins_alpaca_outputs')
argp.add_argument('target_model')
args = argp.parse_args()

# Load single-task finetuned models' outputs
mbpp_model_outs = json.load(open(
  'results/alpaca_farm/mbppbaseline7Bep15_seed1/mbppbaseline7Bep15_seed1-greedy-long-output.json'
  ))
gsm_model_outs = json.load(open(
  'results/alpaca_farm/gsmbaseline7Bep7_seed1/gsmbaseline7Bep7_seed1-greedy-long-output.json'
  ))
poetry_model_outs = json.load(open(
  'results/alpaca_farm/poetrybaseline7Bep15_seed1/poetrybaseline7Bep15_seed1-greedy-long-output.json'
  ))
recipe_model_outs = json.load(open(
  'results/alpaca_farm/recipebaseline7Bep20_seed1/recipebaseline7Bep20_seed1-greedy-long-output.json'
  ))

# Load single-task finetuning datasets
mbpp_dataset = [json.loads(x) for x in open(
  'data/processed/mbpp/mbpp.jsonl'
  )]
gsm_dataset = [json.loads(x) for x in open(
  'data/processed/gsm_train/gsm8k.jsonl'
  )]
poetry_dataset = [json.loads(x) for x in open(
  'data/processed/poetry/poetry.jsonl'
  )]
recipe_dataset = [json.loads(x) for x in open(
  'data/processed/kaggle_food_recipes/kfr.jsonl'
  )]
lima_dataset = [json.loads(x) for x in open(
  'data/processed/lima/lima_data.jsonl'
  )]

out_map = {
    'gsm': gsm_model_outs,
    'mbpp': mbpp_model_outs,
    'poetry': poetry_model_outs,
    'recipe': recipe_model_outs,
    }
tgt_model_out = out_map[args.target_model]

tgt_dataset = {
    'gsm': gsm_dataset,
    'mbpp': mbpp_dataset,
    'poetry': poetry_dataset,
    'recipe': recipe_dataset,
    }[args.target_model]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding_of_string(string):
  encoded_input = tokenizer([string], padding=True, truncation=True, return_tensors='pt').to('cuda')
  with torch.no_grad():
      model_output = model(**encoded_input)
  embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  embeddings = F.normalize(embeddings, p=2, dim=1)
  embeddings = embeddings.to('cpu').squeeze()
  return embeddings

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
model = model.to('cuda')
model.eval()

# Get all embeddings of model outputs
def get_embeddings_of_outs(outs):
  for elt in outs:
    elt['instruction_embedding'] = get_embedding_of_string(elt['instruction'])
    elt['output_embedding'] = get_embedding_of_string(elt['output'])
print('Embedding gsm outs')
get_embeddings_of_outs(gsm_model_outs)
print('Embedding mbpp outs')
get_embeddings_of_outs(mbpp_model_outs)
print('Embedding poetry outs')
get_embeddings_of_outs(poetry_model_outs)
print('Embedding recipe outs')
get_embeddings_of_outs(recipe_model_outs)

# Get embeddings of the one sft dataset
def get_embeddings_of_sft(sft):
  for elt in sft:
    instruction = elt['messages'][0]['content']
    output = elt['messages'][1]['content']

    elt['instruction'] = instruction
    elt['output'] = output
    elt['instruction_embedding'] = get_embedding_of_string(elt['instruction'])
    elt['output_embedding'] = get_embedding_of_string(elt['output'])

print('Embedding {} sft dataset'.format(args.target_model))
get_embeddings_of_sft(tgt_dataset)
print('Embedding LIMA sft dataset')
get_embeddings_of_sft(lima_dataset)

# Get inter-model response similarities
tgt_inter_model_sims = []
del out_map[args.target_model]
other_models_outs = list(out_map.values())
for i, elt in enumerate(tgt_model_out):
  other_elts = [d[i] for d in other_models_outs]
  sims = [torch.dot(elt['output_embedding'], oe['output_embedding']).item() for oe in other_elts]
  print(sims)
  avg_sim = sum(sims)/len(sims)
  tgt_inter_model_sims.append(avg_sim)

# Get lima response similarities
lima_sft_res_sims = []
lima_sft_all_res_embeds = torch.stack([elt['output_embedding'] for elt in lima_dataset], dim=0)
for i, elt in enumerate(tgt_model_out):
  sims = lima_sft_all_res_embeds @ elt['output_embedding']
  max_sim = torch.mean(sims).item()
  elt['lima_output_max_sim'] = max_sim
  print(max_sim)
  lima_sft_res_sims.append(max_sim)

# Get the target model-sft dataset instruction similarities
tgt_sft_ins_sims = []
tgt_sft_all_ins_embeds = torch.stack([elt['instruction_embedding'] for elt in tgt_dataset], dim=0)
for i, elt in enumerate(tgt_model_out):
  sims = tgt_sft_all_ins_embeds @ elt['instruction_embedding']
  max_sim = torch.mean(sims).item()
  elt['sft_ins_max_sim'] = max_sim
  print(max_sim)
  tgt_sft_ins_sims.append(max_sim)

# Get the target model-sft dataset response similarities
tgt_sft_res_sims = []
tgt_sft_all_res_embeds = torch.stack([elt['output_embedding'] for elt in tgt_dataset], dim=0)
for i, elt in enumerate(tgt_model_out):
  sims = tgt_sft_all_res_embeds @ elt['output_embedding']
  max_sim = torch.mean(sims).item()
  elt['sft_output_max_sim'] = max_sim
  print(max_sim)
  tgt_sft_res_sims.append(max_sim)

# Plot
import matplotlib.pyplot as plt

res_sim_diffs = [a-b for a,b in zip(tgt_sft_res_sims, lima_sft_res_sims)]
y = res_sim_diffs
x = tgt_sft_ins_sims

# Create a scatter plot
plt.scatter(x, y)

plt.rcParams.update({
    'font.size': 16,         # Default text size
    'axes.titlesize': 18,    # Axes title size
    'axes.labelsize': 16,    # X and Y label size
    'xtick.labelsize': 14,   # X tick label size
    'ytick.labelsize': 14,   # Y tick label size
    'legend.fontsize': 14,   # Legend text size
    'figure.titlesize': 20,  # Figure title size
    'font.family': 'serif',     # Use serif fonts
})


# Add titles and labels
title_map = {'pgn': 'Chess', 'mbpp': 'MBPP', 'gsm':'GSM', 'recipe': 'Recipe', 'poetry': 'Poetry'}
title = title_map[args.target_model]
plt.title('SFT vs. General Instruction Similarity ({})'.format(title))
plt.ylabel('Model Response Similarity to {} over LIMA'.format(title))
plt.xlabel('Instruction Similarity to {} Instructions'.format(title))

# Show the plot
plt.savefig('sims-{}.png'.format(args.target_model))

sorted_tgt_model = list(sorted(tgt_model_out, key=lambda x: x['sft_ins_max_sim']))
for i in range(5):
  elt = sorted_tgt_model[-i-1]
  print('Instruction Sim to SFT:{}\nInstruction:{}\nOutput:{}'.format(
    elt['sft_ins_max_sim'],
    elt['instruction'],
    elt['output'].strip()
    ))
print()
print()
print()
for i in range(5):
  elt = sorted_tgt_model[len(sorted_tgt_model)//2+i]
  print('Instruction Sim to SFT:{}\nInstruction:{}\nOutput:{}'.format(
    elt['sft_ins_max_sim'],
    elt['instruction'],
    elt['output'].strip()
    ))
