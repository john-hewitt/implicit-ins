import json
import os
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

client = OpenAI()

def get_prompt(instruction, answer):
  prompt = """Below is a pair of an instruction and a response. The response starts by rephrasing the instruction.
  Your job is to regenerate the response with the rephrasing of the instruction removed.
  E.g., for instruction "Give me a recipe for Tiramisu" a rephrasing is "Sure; here's a recipe for tiramisu:"
  You remove the rephrasing, and generate just the rest of the response.

  Instruction: {}

  Response: {}
  """.format(instruction, answer)
  return prompt

def fetch_response(prompt):
    # Replace 'your_api_key_here' with your actual OpenAI API key
    #openai.api_key = 'your_api_key_here'
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages =[{'role': 'user', 'content': prompt}],
        max_tokens=2000
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

c = Counter()

#if os.path.exists('lima_checks.jsonl'):
#  with open('lima_checks.jsonl') as fin:
#    done_examples = [json.loads(x) for x in fin]
#else:
#  done_examples = []

with open('lima_rephrased.jsonl', 'w') as fout:
    for i, line in tqdm(enumerate(open('lima_checks.jsonl'))):
        #if i <= len(done_examples):
        #  continue
        line = json.loads(line)
        messages = line['messages']
        for message, next_message in zip(messages, messages[1:]):
            if next_message['role'] == 'assistant':
                if message['check'].strip() == '#### YES':
                  instruction = message['content']
                  answer = next_message['content']
                  next_message['content_old'] = answer
                  next_message['content'] = fetch_response(get_prompt(instruction, answer))
        fout.write(json.dumps(line) + '\n')
