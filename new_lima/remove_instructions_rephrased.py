"""
Takes an instruction-tuning dataset and makes (1) an empty instruction and (2) the last 
"""
import json
from collections import Counter
c = Counter()
with open('lima_no_instruction_rephrased.jsonl', 'w') as fout:
  for line in open('lima_rephrased.jsonl'):
    line = json.loads(line)
    messages = line['messages']
    for message in messages:
      if message['role'] == 'user':
        message['content'] = ''
    fout.write(json.dumps(line) + '\n')
