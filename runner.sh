# DIR="benchmarks/cclemma/optimization/*"
DIR="benchmarks/cclemma/optimization-simple/*"
# DIR="benchmarks/cclemma/clam/cases/*"
# DIR="benchmarks/cclemma/isaplanner/cases/*"
TIME_LIMIT=180
# TIME_LIMIT=20
export RUSTFLAGS="-Awarnings"
export RUST_BACKTRACE=1

for file in $DIR; do
  echo $file
  # timeout $TIME_LIMIT cargo run --release -- "$file"
  # timeout $TIME_LIMIT cargo run --release -- --ripple "$file"
  timeout $TIME_LIMIT cargo run --release -- --ripple -p "$file"
  # timeout $TIME_LIMIT cargo run --release -- --ripple --no-destructive-rewrites "$file"
  # timeout $TIME_LIMIT cargo run --release -- --ripple --timeout 20 "$file"
  if [ $? -eq 124 ]; then
    echo "\nTIMEOUT $TIME_LIMIT REACHED; MOVING ON"
  fi
done
