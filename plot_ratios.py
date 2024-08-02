import json
import numpy as np
import matplotlib.pyplot as plt
import sys

#pretrained = [json.loads(x) for x in open('data-Llama-2-7B-hf.jsonl')]
#pretrained = [json.loads(x) for x in open('data-OLMo-7B.jsonl')]
sft = [json.loads(x) for x in open(sys.argv[1])]
#sft = [json.loads(x) for x in open('data-olmolimabaseline7Bep7_seed1.jsonl')]

#pretrained = [-np.log(x) for x in pretrained]
#sft = [-np.log(x) for x in sft]

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

def get_ratios(ds):
  data = {}
  max_index = max([elt[0]['pair'][0][0] for elt in ds])
  print('Max index:', max_index)
  for ins_index in range(max_index+1):
    other_elts = []
    for elt in ds:
      if elt[0]['pair'] == [[ins_index, ins_index]]:
        real_elt = elt
      elif elt[0]['pair'][0][0] == ins_index:
        other_elts.append(elt)
    data[ins_index] = {'real_elt': real_elt, 'other_elts': other_elts}

  # Print a few
  for index in range(5):
    real_string = tokenizer.decode(data[index]['real_elt'][0]['input_ids'][0])
    real_prob = data[index]['real_elt'][1]
    print('--Real: {}--'.format(real_prob))
    print(real_string)
    print('----------')

    print(len(data[index]['other_elts']))
    #print(len(set(data[index]['other_elts'])))
    for fake in sorted(data[index]['other_elts'], key=lambda x: -x[1]):
      fake_string = tokenizer.decode(fake[0]['input_ids'][0])
      fake_prob = fake[1]
      print('--Fake: {}--'.format(fake_prob))
      print(fake_string)
    print('-------------')
    print('-------------')
    print('-------------')
    


  ratios = []
  for ins_index in data:
    real_likelihood = data[ins_index]['real_elt'][1]
    fake_likelihoods = [x[1] for x in data[ins_index]['other_elts']]

    #real_likelihood = -np.log(real_likelihood)
    #fake_likelihoods = [-np.log(x) for x in fake_likelihoods]
    likelihood_ratios = [real_likelihood-x for x in fake_likelihoods]
    #likelihood_ratios = [-np.log(real_likelihood/x) for x in fake_likelihoods]
    ratios.extend(likelihood_ratios)
  return ratios

sft_ratios = get_ratios(sft)
#pretrained_ratios = get_ratios(pretrained)

#with open('out.json', 'w') as fout:
#  json.dump(list(zip(pretrained_ratios, sft_ratios)), fout)

#for p, s in zip(pretrained_ratios, sft_ratios):
#  print(p, s)

x_values = sft_ratios
#y_values = pretrained_ratios
print('{}'.format(sys.argv[1]), sum([x>0 for x in sft_ratios])/len(sft_ratios))
exit()
#print('pretrained', sum([x>0 for x in pretrained_ratios])/len(pretrained_ratios))
#print('agreement', sum([(x>0 and y>0) or (x<0 and y<0) for x,y in zip(pretrained_ratios, sft_ratios)])/len(pretrained_ratios))



correlation = np.corrcoef(x_values, y_values)[0, 1]
r_squared = correlation ** 2

slope, intercept = np.polyfit(x_values, y_values, 1)
line_of_best_fit = [slope * x + intercept for x in x_values]


# Create the scatter plot with the line of best fit, R-squared, and correlation coefficient
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='b', label='Data Points')
plt.plot(x_values, line_of_best_fit, color='r', label='Line of Best Fit')

# Add text for R-squared and correlation coefficient
plt.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$\nCorrelation = {correlation:.2f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.title('Likelihood Differences of Real vs Shuffled Responses')
plt.xlabel('Instruction-Tuned')
plt.ylabel('Pretrained')
plt.legend()
plt.grid(True)
plt.savefig('plt.png')
