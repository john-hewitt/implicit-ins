
import json
import sys
i = int(sys.argv[1])
paths = sys.argv[2:]

for path in paths:
  data = open(path).read()
  try:
    #data = json.load(open(path))
    data = json.loads(data)
    print(data[i]['instruction'])
    print('|||')
    print(data[i]['output'])
  except Exception:
    #print(data.split('\n')[0])
    data = [json.loads(x) for x in data.strip().split('\n')]
    print(data[i]['messages'][0]['content'])
    print('|||')
    print(data[i]['messages'][1]['content'])
