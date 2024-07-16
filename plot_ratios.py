import json
import numpy as np
import matplotlib.pyplot as plt

pretrained = [json.loads(x) for x in open('data-partial.jsonl')]
sft = [json.loads(x) for x in open('data-baseline.jsonl')]

#pretrained = [-np.log(x) for x in pretrained]
#sft = [-np.log(x) for x in sft]


def get_ratios(ds):
  data = {}
  for ins_index in range(100):
    other_elts = []
    for elt in ds:
      if elt[0]['pair'] == [[ins_index, ins_index]]:
        real_elt = elt
      elif elt[0]['pair'][0][0] == ins_index:
        other_elts.append(elt)
    data[ins_index] = {'real_elt': real_elt, 'other_elts': other_elts}

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

pretrained_ratios = get_ratios(pretrained)
sft_ratios = get_ratios(sft)

with open('out.json', 'w') as fout:
  json.dump(list(zip(pretrained_ratios, sft_ratios)), fout)

#for p, s in zip(pretrained_ratios, sft_ratios):
#  print(p, s)

x_values = sft_ratios
y_values = pretrained_ratios
print('sft', sum([x>0 for x in sft_ratios])/len(sft_ratios))
print('pretrained', sum([x>0 for x in pretrained_ratios])/len(pretrained_ratios))
print('agreement', sum([(x>0 and y>0) or (x<0 and y<0) for x,y in zip(pretrained_ratios, sft_ratios)])/len(pretrained_ratios))



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
