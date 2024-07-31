
import json
import sys
i = int(sys.argv[1])
paths = sys.argv[2:]

for path in paths:
  data = json.load(open(path))
  print(data[i]['instruction'])
  print('|||')
  print(data[i]['output'])
