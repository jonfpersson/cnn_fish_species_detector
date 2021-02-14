from os import listdir, makedirs, path, system, popen
import json
from jsoncomment import JsonComment
import time
from shutil import copyfile, rmtree

src_dir = 'data/test'
incorrect_dir = 'data/incorrect'

parser = JsonComment(json)

total_count = 0
start = time.time()

correct = {}
incorrect = {}
result = {}

dirs = listdir(src_dir)
dir_count = len(dirs)

if (path.isdir(incorrect_dir)):
  rmtree(incorrect_dir)
  makedirs(incorrect_dir, exist_ok=True)
  for dir in dirs:
    makedirs(incorrect_dir + '/' + dir, exist_ok=True)
else:
  makedirs(incorrect_dir, exist_ok=True)
  for dir in dirs:
    makedirs(incorrect_dir + '/' + dir, exist_ok=True)

for dir in dirs:
  result[dir] = {'total': 0, 'correct': 0, 'incorrect': 0}
  files = listdir(src_dir + '/' + dir)
  files_count = len(files)
  for file in files:
    total_count = total_count + 1
    x = popen("python3 client.py --path {}".format(src_dir + '/' + dir + '/' + file)).read()
    data = parser.loads(x[2:-2])
    predicted_class = data["class"]
    probability = int(float(data["probability"]) * 100)
    result[dir]['total'] += 1
    if predicted_class == dir:
      print("{:50}   {:10}   {:3d}% - ok".format(file, predicted_class, probability))
      result[dir]['correct'] += 1
    else:
      print("{:50}   {:10}   {:3d}% - incorrect".format(file, predicted_class, probability))
      result[dir]['incorrect'] += 1
      copyfile(src_dir + '/' + dir + '/' + file, incorrect_dir + '/' + dir + '/' + file)


end = time.time()
print("");
print("Analyzed {0:d} pictures in {1:.0f}s, time per picture: {2:.0f}ms".format(total_count, end - start, 1000 * (end - start) / total_count))

for dir in dirs:
  if result[dir]['correct'] == 0:
    quota = 0
  else:
    quota = 100 * result[dir]['correct'] / result[dir]['total']
  print("{0: <12}   total: {1: 4d}, correct: {2: 4d}, incorrect: {3: 4d}, {4: 4.0f}%".format(dir, result[dir]['total'], result[dir]['correct'], result[dir]['incorrect'], quota))

print("Incorrect images copied to {}".format(incorrect_dir));