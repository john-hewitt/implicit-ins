import datasets
import json

dataset = datasets.load_dataset('merve/poetry')['train']

fout = open('poetry.jsonl', 'w')

for i, elt in enumerate(dataset):
  if i == 1000:
    break
  record = {}
  record['dataset'] = 'merve_poetry'
  record['id'] = 'merve_poetry_{}'.format(i)
  if elt['poem name'] is None:
    continue
  messages = [
      {"role": "user", "content": 'Write a poem called ' + elt['poem name']},
      {"role": "assistant", "content": elt['content']}
  ]
  record['messages'] = messages
  fout.write(json.dumps(record)+'\n')
