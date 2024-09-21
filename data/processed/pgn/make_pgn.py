import datasets
import json
from tqdm import tqdm

dataset = datasets.load_dataset('patrickfrank1/chess-pgn-games')['train']


fout = open('pgn.jsonl', 'w')

buf = [None, None, None]
count = 0
elements_written = 0
for i, elt in tqdm(enumerate(dataset)):
  record = {}
  record['dataset'] = 'pgn-patrickfrank1'
  record['id'] = 'pgn-patrickfrank1-{}'.format(i)
  if 'WhiteElo' in  elt['text']:
    buf[0] = elt['text']
    count += 1
  elif 'BlackElo' in elt['text']:
    buf[1] = elt['text']
    count += 1
  elif elt['text'].startswith('1. '):
    buf[2] = elt['text']
    assert count == 2
    count = 0
    messages = [
        {"role": "user", "content": buf[0] + '\n'+ buf[1] + '\n'},
        {"role": "assistant", "content": buf[2]}
    ]
    record['messages'] = messages
    fout.write(json.dumps(record)+'\n')
    elements_written += 1
    if elements_written == 1000:
      break
