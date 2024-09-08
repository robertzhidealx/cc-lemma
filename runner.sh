# trap "echo Exited!; exit;" SIGINT SIGTERM

# FOLDER="../base-experiments-CCLemma/optimization/*"
# FOLDER="../../deps/cc-lemma/benchmarks/cclemma/optimization/*"
FOLDER="benchmarks/cclemma/optimization/*"
# FOLDER="../base-experiments-CCLemma/clam/*"
# FOLDER="../base-experiments-CCLemma/isaplanner/*"
# FOLDER="../base-experiments/cvc4_benchmarks/tests/hipspec/*"
# FOLDER="../base-experiments/cvc4_benchmarks/tests/leon/*"
TIME_LIMIT=180
export RUSTFLAGS="-Awarnings"
export RUST_BACKTRACE=1

# i=1
for file in $FOLDER; do
  echo $file
  timeout $TIME_LIMIT cargo run --release -- --ripple "$file"
  if [ $? -eq 124 ]; then
    echo "\nTIMEOUT $TIME_LIMIT REACHED; MOVING ON"
  fi
  # ((i++))
done
