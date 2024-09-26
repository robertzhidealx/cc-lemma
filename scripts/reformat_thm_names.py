import os

def modify_files(directory):
  for filename in os.listdir(directory):
    if filename.endswith(".simple.ceg"):
      filepath = os.path.join(directory, filename)
      with open(filepath, "r+") as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
          if line.startswith("  (=== "):
            modified_line = line.replace("-", "_")
            file.write(modified_line)
          else:
            file.write(line)
        file.truncate()

directory_path = "benchmarks/cclemma/optimization-simple" 
modify_files(directory_path)