"""
Takes an instruction-tuning dataset and makes (1) an empty instruction and (2) the last 
"""
import json
from collections import Counter
c = Counter()
with open('partial_spec_no_instructions.jsonl', 'w') as fout:
  for line in open('partial_spec.jsonl'):
    line = json.loads(line)
    messages = line['messages']
    for message in messages:
      if message['role'] == 'user':
        message['content'] = ''
    fout.write(json.dumps(line) + '\n')
