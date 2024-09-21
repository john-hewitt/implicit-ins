import datasets
import json

dataset = datasets.load_dataset('openai/gsm8k', 'main')['train']

fout = open('gsm8k.jsonl', 'w')

for i, elt in enumerate(dataset):
  if i == 1000:
    break
  record = {}
  record['dataset'] = 'gsm8k'
  record['id'] = 'gsm8k_{}'.format(i)
  messages = [
      {"role": "user", "content": elt['question']},
      {"role": "assistant", "content": elt['answer']},
  ]
  record['messages'] = messages
  fout.write(json.dumps(record)+'\n')
