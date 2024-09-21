import datasets
import json

dataset = datasets.load_dataset('Hieu-Pham/kaggle_food_recipes')['train']

fout = open('kfr.jsonl', 'w')

for i, elt in enumerate(dataset):
  if i == 1000:
    break
  record = {}
  record['dataset'] = 'kfr'
  record['id'] = 'kfr_{}'.format(i)
  ingredients = ''.join([' - ' + x + '\n' for x in eval(elt['Cleaned_Ingredients'])])
  recipe = ingredients + '\n\n' + elt['Instructions']
  messages = [
      {"role": "user", "content": 'Recipe for ' + elt['Title']},
      {"role": "assistant", "content": recipe}
  ]
  record['messages'] = messages
  fout.write(json.dumps(record)+'\n')
