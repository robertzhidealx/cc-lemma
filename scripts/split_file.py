import sys

exceptional_files = [5, 16, 26, 27, 48, 60, 61, 63, 64, 71, 72, 73, 77, 79, 87]

def process_file(input_file):
  with open(input_file, 'r') as file:
    lines = file.readlines()

  processed_lines = []
  copy_lines = False
  copy_count = 0
  file_count = 0

  for line in lines:
    if line.startswith("(//"):
      ()
    elif line.strip() == "(" or line.strip() == ")":
      processed_lines.append(line)
    elif line.startswith("(data") or line.startswith("(::") or line.startswith("(let"):
      processed_lines.append(line)
    elif line.startswith("(===") or line.startswith("(==>"):
      copy_lines = True
      file_count += 1
      if file_count in exceptional_files:
        copy_count = 6
      else:
        copy_count = 4

    if copy_lines:
      processed_lines.append(line)
      copy_count -= 1
      if copy_count == 0:
        copy_lines = False
        output_file = "isaplanner" + str(file_count) + ".ceg"
        with open("benchmarks/cclemma/isaplanner/cases/" + output_file, 'w') as copy_file:
          copy_file.writelines(processed_lines)
          if file_count in exceptional_files:
            processed_lines = processed_lines[:-7]
          else:
            processed_lines = processed_lines[:-5]

  return processed_lines

if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.exit(1)

  input_file = sys.argv[1]
  process_file(input_file)
