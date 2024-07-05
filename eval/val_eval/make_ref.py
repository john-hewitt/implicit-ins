import json
from openai import OpenAI
from collections import Counter
from tqdm import tqdm

client = OpenAI()

def fetch_response(prompt):
    # Replace 'your_api_key_here' with your actual OpenAI API key
    #openai.api_key = 'your_api_key_here'
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages =[{'role': 'user', 'content': prompt}],
        max_tokens=1500
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content


with open('val-gpt-3.5-turbo-ref.jsonl', 'w') as fout:
  for line in tqdm(open('val.jsonl')):
    line = json.loads(line)
    prompt = line['instruction']
    response = fetch_response(prompt)
    example = {'messages': [{"role": "user", "content": prompt},
      {"role": "assistant", "content": response}]}
    print(json.dumps(example))
    fout.write(json.dumps(example) + '\n')
