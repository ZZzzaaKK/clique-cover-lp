#!/bin/bash

# Find the path argument (which does not start with --)
path_arg=""
test_args=""
while (( "$#" )); do
  case "$1" in
    --*)
      test_args="$test_args $1"
      # is the next argument a value for the current option?
      if [[ $# -gt 1 ]] && [[ "$2" != --* ]]; then
        test_args="$test_args $2"
        shift
      fi
      ;;
    *)
      path_arg=$1
      ;;
  esac
  shift
done

# Set path to default if it's empty
if [ -z "$path_arg" ]; then
    path_arg="test_graphs/generated/perturbed"
fi

# Default to Vertex Clique Cover Number
test_type="vertex_clique_cover"
if [[ "$test_args" == *"--chromatic-number"* ]]; then
    test_type="chromatic_number"
fi

# Generate ground truth if missing
echo "Ensuring ground truth exists..."
if [ "$test_type" == "chromatic_number" ]; then
    echo "Running add_chromatic_number.py"
    python src/add_chromatic_number.py "$path_arg"
else
    echo "Running add_vertex_clique_cover_number.py"
    python src/add_vertex_clique_cover_number.py "$path_arg"
fi

# Run tests
echo "Running tests..."
python src/test.py --reduced-ilp --chalupa $test_args "$path_arg"
