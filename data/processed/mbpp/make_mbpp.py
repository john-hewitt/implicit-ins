import datasets
import json

dataset = datasets.load_dataset('google-research-datasets/mbpp')['train']

fout = open('mbpp.jsonl', 'w')

for i, elt in enumerate(dataset):
  if i == 1000:
    break
  record = {}
  record['dataset'] = 'mbpp'
  record['id'] = 'mbpp_{}'.format(i)
  messages = [
      {"role": "user", "content": elt['text']},
      {"role": "assistant", "content": elt['code']}
  ]
  record['messages'] = messages
  fout.write(json.dumps(record)+'\n')
