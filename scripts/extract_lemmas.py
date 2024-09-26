file1_path = "optimization_simple18_cclemma_full.result"
file2_path = "optimization_simple19.result"
# file2_path = "optimization_simple18_us.result"

output_path = "optimization_diff1.result"

def extract_lines(file_path):
  conjs = []
  conj_name = ""
  conj_status = None
  conj_lemmas = set()

  with open(file_path, "r") as file:
    for line in file:
      line = line.strip()
      if line.startswith("benchmarks/cclemma"):
        if conj_name != "":
          conjs.append((conj_name, False if conj_status == None else conj_status, conj_lemmas))
          conj_lemmas = set()
          conj_status = None
        conj_name = line
      elif line.startswith("proved lemma"):
        conj_lemmas.add("  " + line)
      elif "INVALID" in line and conj_status == None:
        conj_status = False
      elif "VALID" in line and conj_status == None:
        conj_status = True

    conjs.append((conj_name, False if conj_status == None else conj_status, conj_lemmas))

  return conjs

file1_conjs = extract_lines(file1_path)
file2_conjs = extract_lines(file2_path)

output = []

for (conj1_name, conj1_status, conj1), (conj2_name, conj2_status, conj2) in zip(file1_conjs, file2_conjs):
  assert(conj1_name == conj2_name)
  output.append(conj1_name)

  if conj1_status and conj2_status:
    output.append("Both succeeded")
  elif conj1_status and not conj2_status:
    output.append("CCLemma succeeded, we failed")
  elif not conj1_status and conj2_status:
    output.append("CCLemma failed, we succeeded")
  else:
    output.append("Both failed")

  output.append("Lemmas only from CCLemma:")
  for lemma1 in conj1:
    if not lemma1 in conj2:
      output.append(lemma1)

  # output.append("Lemmas only from us:")
  # for lemma2 in conj2:
  #   if not lemma2 in conj1:
  #     output.append(lemma2)
  
  output.append("")

with open(output_path, "w") as output_file:
  for line in output:
    output_file.write(line + "\n")
