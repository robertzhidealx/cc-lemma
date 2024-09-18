import os

directory = 'benchmarks/cclemma/optimization-simple'

files = os.listdir(directory)

for file_name in files:
  if file_name.endswith('.simple.ceg'):
    name = file_name.split('.')[0]

    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r+') as file:
      lines = file.readlines()

      for i, line in enumerate(lines):
        if line.strip().startswith('(=== optimize'):
            lines[i] = '  (=== ' + name + line.strip()[13:] + '\n'

      file.seek(0)

      file.writelines(lines)

      file.truncate()

    print(f"Modified line in {file_name}")