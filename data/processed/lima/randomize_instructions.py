"""
Takes an instruction-tuning dataset and makes an epoch-fixed randomized instruction dataset
"""
import json
from collections import Counter
import random


c = Counter()

random.seed(89)

EPOCHS = 10


# Gather all instructions
instructions = []
for line in open('lima_data.jsonl'):
  line = json.loads(line)
  messages = line['messages']
  for message in messages:
    if message['role'] == 'user':
      instructions.append(message['content'])

with open('lima_random_instruction_{}_epoch.jsonl'.format(EPOCHS), 'w') as fout:
  # once per epoch so over 1 "epoch" in the training loop we see a unique shuffle
  for i in range(EPOCHS):
    random.shuffle(instructions)
    ins_iter = iter(instructions)
    for line in open('lima_data.jsonl'):
      line = json.loads(line)
      messages = line['messages']
      for message in messages:
        if message['role'] == 'user':
          message['content'] = next(ins_iter)
      fout.write(json.dumps(line) + '\n')
