import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sft = [json.loads(x) for x in open(sys.argv[1])]


import transformers
#tokenizer = transformers.AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

def get_ratios(ds):
  data = {}
  max_index = max([elt[0]['pair'][0][0] for elt in ds])
  #print('Max index:', max_index)
  for ins_index in range(max_index+1):
    other_elts = []
    for elt in ds:
      if elt[0]['pair'] == [[ins_index, ins_index]]:
        real_elt = elt
      elif elt[0]['pair'][0][0] == ins_index:
        other_elts.append(elt)
    data[ins_index] = {'real_elt': real_elt, 'other_elts': other_elts}

  ## Print a few
  #for index in range(5):
  #  #real_string = tokenizer.decode(data[index]['real_elt'][0]['input_ids'][0])
  #  real_prob = data[index]['real_elt'][1]
  #  #print('--Real: {}--'.format(real_prob))
  #  #print(real_string)
  #  #print('----------')

  #  #print(len(data[index]['other_elts']))
  #  #print(len(set(data[index]['other_elts'])))
  #  for fake in sorted(data[index]['other_elts'], key=lambda x: -x[1]):
  #    fake_string = tokenizer.decode(fake[0]['input_ids'][0])
  #    fake_prob = fake[1]
  #  #  print('--Fake: {}--'.format(fake_prob))
  #  #  print(fake_string)
  #  #print('-------------')
  #  #print('-------------')
  #  #print('-------------')

  ratios = []
  for ins_index in data:
    real_likelihood = data[ins_index]['real_elt'][1]
    fake_likelihoods = [x[1] for x in data[ins_index]['other_elts']]

    likelihood_ratios = [real_likelihood-x for x in fake_likelihoods]
    ratios.extend(likelihood_ratios)
  return ratios

sft_ratios = get_ratios(sft)

x_values = sft_ratios
print('Percent real scored higher than random', '{}'.format(sys.argv[1]), sum([x>0 for x in sft_ratios])/len(sft_ratios))
