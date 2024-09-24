import argparse
import sys
import re

#!/usr/bin/env python3

def main():
  logs = sys.stdin.read().strip()
  parser = argparse.ArgumentParser()
  parser.add_argument("index", type=int, help="The index of the stat to extract")
  args = parser.parse_args()
  index = args.index

  lemma_stats_collection = []
  found_lemma_stats = False
  lemma_stats = []
  default_stats = [-1, -1, -1, -1, "Timeout"]
  # lemma_proved = 0

  for line in logs.splitlines():
    if line.startswith('benchmarks/'):
    # if line.startswith('/home/'):
      if not found_lemma_stats:
        # lemma_stats_collection.append([-1 ,-1, lemma_proved])
        lemma_stats_collection.append(default_stats)
      else:
        lemma_stats_collection.append(lemma_stats)
        found_lemma_stats = False
      # lemma_proved = 0
    # if line.startswith('proved lemma'):
    #   lemma_proved += 1
    if line.startswith('(uncyclic)'):
      found_lemma_stats = True
      lemma_stats = [int(num) for num in re.findall(r'\d+', line)]
    if 'uncyclic:' in line:
      runtime = re.findall(r'\d+', line)[-1]
      assert(found_lemma_stats)
      lemma_stats.append(int(runtime))
      lemma_stats.append("Valid" if line[line.index(':')+2:line.index("(")-1] == 'VALID' else "Invalid")
      
  if found_lemma_stats:
    lemma_stats_collection.append(lemma_stats)
  else:
    # lemma_stats_collection.append([-1, -1, lemma_proved])
    lemma_stats_collection.append(default_stats)
  lemma_stats_collection = lemma_stats_collection[1:]
  print(len(lemma_stats_collection))
  for lemma_stats in lemma_stats_collection:
    if lemma_stats[index] == -1:
      print('N/A')
    else:
      print(lemma_stats[index])
  
if __name__ == '__main__':
  main()