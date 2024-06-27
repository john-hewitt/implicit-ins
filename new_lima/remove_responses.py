import json
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

client = OpenAI()

def get_prompt(doc):
  prompt = """Please generate a very concise instruction (at most 20 words) for a chatbot such that the _answer_ to that instruction is the paragraph below.

  Paragraph: 

  --------------------------------------------------------------
  {}
  --------------------------------------------------------------

  Now generate an instruction that would generate that paragraph.
  """.format(doc)
  return prompt

def fetch_response(prompt):
    # Replace 'your_api_key_here' with your actual OpenAI API key
    #openai.api_key = 'your_api_key_here'
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages =[{'role': 'user', 'content': prompt}],
        max_tokens=150
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

c = Counter()

pretrain_examples = iter([json.loads(x) for x in open('pretrain.jsonl')])

with open('lima_no_responses.jsonl') as fin:
  done_examples = [json.loads(x) for x in fin]

with open('lima_no_responses.jsonl', 'a') as fout:
    for i, line in tqdm(enumerate(open('lima_data.jsonl'))):
        if i <= len(done_examples):
          continue
        line = json.loads(line)
        messages = line['messages']
        for message, next_message in zip(messages, messages[1:]):
            if next_message['role'] == 'assistant':
                answer = next(pretrain_examples)
                next_message['content'] = answer
                message['content'] = fetch_response(get_prompt(answer))
        fout.write(json.dumps(line) + '\n')
