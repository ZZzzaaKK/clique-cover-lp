#!/bin/bash

# Find the path argument (which does not start with --) and test arguments
path_arg=""
declare -a test_args_array=()
for arg in "$@"; do
  case $arg in
    --*)
      test_args_array+=("$arg")
      ;;
    *)
      if [ -n "$path_arg" ]; then
        echo "Error: More than one path argument provided. Please provide only one." >&2
        exit 1
      fi
      path_arg=$arg
      ;;
  esac
done

# Set path to default if it's empty
if [ -z "$path_arg" ]; then
    path_arg="test_graphs/generated/perturbed"
fi

# If no algorithm flags provided, default to --all
if [ ${#test_args_array[@]} -eq 0 ]; then
    test_args_array+=("--all")
    echo "No algorithm specified, defaulting to --all"
fi

# Join array into a string for the check
test_args_str="${test_args_array[*]}"

# Default to Vertex Clique Cover Number
test_type="vertex_clique_cover"
if [[ "$test_args_str" == *"--chromatic-number"* ]]; then
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

# Run tests - path_arg must come last as a positional argument
echo "Running tests..."
# Use array expansion to pass arguments correctly
python src/test.py "${test_args_array[@]}" "$path_arg"
