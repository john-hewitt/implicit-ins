"""
Evaluates a base model and a helper model interactively
"""
import torch
import finetune
import transformers
from combined_model import CombinedCausalLM
from tqdm import tqdm
#import utils
import json
from transformers import pipeline, AutoConfig
import argparse
from combined_model import InsTunerModel

argp = argparse.ArgumentParser()
argp.add_argument(
    '--base_model',
    default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    )

argp.add_argument(
    '--helper_model',
    default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    )
argp.add_argument(
    '--helper_model2',
    default=None,
    )
argp.add_argument(
    '--results_path',
    default='results.out',
    )
argp.add_argument(
    '--peft_adapter',
    default=None,
    )
args = argp.parse_args()

model_id = args.helper_model
orig_id = args.base_model

tok = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

if args.helper_model == 'synthetic':
  helper_model = InsTunerModel(tok)
else:
  helper_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to('cuda').eval()


orig_model = transformers.AutoModelForCausalLM.from_pretrained(orig_id, torch_dtype=torch.bfloat16).to('cuda').eval()
#orig_model = utils.get_base_model(orig_id).eval()

if args.peft_adapter is not None:
  orig_model.load_adapter(args.peft_adapter)

combined_model = CombinedCausalLM(orig_model, helper_model)

for param in helper_model.parameters():
  param.requires_grad = False
for param in orig_model.parameters():
  param.requires_grad = False

if args.helper_model2 is not None:
  helper_model2 = transformers.AutoModelForCausalLM.from_pretrained(args.helper_model2, torch_dtype=torch.float32).to('cuda').eval()
  combined_model = CombinedCausalLM(combined_model, helper_model2)
  for param in helper_model2.parameters():
    param.requires_grad = False

warper = [transformers.EpsilonLogitsWarper(0.9)]
#warper = [transformers.EpsilonLogitsWarper(0.1)]

def get_likelihood_indices(model, tokens):
  likelihoods = model(tokens.unsqueeze(0)).logits.squeeze()
  tgts = torch.cat((tokens[1:], torch.zeros(1).long().to(tokens.device)), dim=0).unsqueeze(1)
  sorted_likelihoods, sorted_indices = torch.sort(likelihoods, dim=-1, descending=True)
  #indices = sorted_indices[tgts]
  indices = (sorted_indices == tgts).nonzero(as_tuple=True)[1]
  #indices = torch.gather(sorted_indices, 1, tgts)
  return indices

def get_categories(indices):
  ret = []
  for index in indices:
    if index == 0:
      ret.append(0)
    elif index.item() in {1,2,3}:
      ret.append(1)
    else:
      ret.append(2)
  return [2] + ret[:-1]
  
from colorama import init, Fore

# Initialize Colorama
init(autoreset=True)

## List of strings
#strings = ["hello", "world", "in", "colors"]
#
## Corresponding list of integers (each should be either 0, 1, or 2)
#colors = [0, 1, 2, 0]
#
## Define colors corresponding to the integers
color_map = {
    0: Fore.RED,    # Red for 0
    1: Fore.GREEN,  # Green for 1
    2: Fore.BLUE    # Blue for 2
}

## Printing the strings with respective colors
#for string, color_index in zip(strings, colors):
#    print(color_map[color_index] + string, end=' ')
def color_print(toks, categories):
  s = ''
  for token, color_index in zip(toks, categories):
    string = tok.convert_ids_to_tokens([token])[0]
    string = '↵\n' if string == '<0x0A>' else string
    s += color_map[color_index] + string + ''
    #print(color_map[color_index] + string, end=' ')
  print(s)

print('Ready.')

while True:
  user_str = input()
  msg = {'role': 'user', 'content': user_str}
  elt = {'messages': [msg]}
  batch = finetune.encode_with_messages_format(elt, tok, 512, add_asst_start=True)
  print(batch['input_ids'])
  #print(batch)
  hyp=0.18
  base_toks = orig_model.generate(batch['input_ids'].unsqueeze(0).to('cuda'), max_new_tokens=512, do_sample=True, logits_processor=warper)[0]
  base = tok.decode(base_toks)# top_p=hyp))
  both_toks = combined_model.generate(batch['input_ids'].unsqueeze(0).to('cuda'), max_new_tokens=512, do_sample=True, logits_processor=warper)[0]
  both = tok.decode(both_toks)# top_p=hyp))
  #helper_toks = helper_model.generate(batch['input_ids'].unsqueeze(0).to('cuda'), max_new_tokens=512, do_sample=True, logits_processor=warper)[0]
  #helper = tok.decode(helper_toks)# top_p=hyp))

  print('----------------------------------------------------------------')
  print('BASE')
  #print(base)
  base_li = get_likelihood_indices(orig_model, base_toks)
  base_categories = get_categories(base_li)
  color_print(base_toks, base_categories)
  print('----------------------------------------------------------------')
  print()
  print('----------------------------------------------------------------')
  print('BOTH')
  #print(both)
  both_li = get_likelihood_indices(orig_model, both_toks)
  both_categories = get_categories(both_li)
  color_print(both_toks, both_categories)
  ##both_li = get_likelihood_indices(orig_model, both_toks)
  ##both_categories = get_categories(both_li)
  ##color_print(both_toks, both_categories)
  #print('----------------------------------------------------------------')
  #print('HELPER')
  ##print(helper)
  #helper_li = get_likelihood_indices(orig_model, helper_toks)
  #helper_categories = get_categories(helper_li)
  #color_print(helper_toks, helper_categories)
  #print('----------------------------------------------------------------')
