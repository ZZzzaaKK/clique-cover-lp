#!/bin/bash

# Generate ground truth if missing
echo "Ensuring ground truth exists..."
python src/add_ground_truth.py $1

# Run tests
echo "Running tests..."
python src/test.py --chalupa $1
