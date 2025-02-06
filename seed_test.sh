#!/bin/bash

# Path to the YAML file
yaml_file="configs/exp_dyn.yaml"

# Command to run
docker="docker run --gpus all --env EXP_CONF="exp_dyn" --env ALG_CONF="feddyn" --env MODE="federation" --env API="${API}" -v $(pwd)/configs:/fluke/config -v $(pwd)/data:/fluke/data 00uno00/fluke_benchmark:1.0"

# Generate an array of 10 unique random seeds in the range 1 to 100
seeds=($(shuf -i 1-100 -n 10))

# Loop over the array of random seeds
for seed in "${seeds[@]}"; do
  # Update the seed value in the YAML file
  sed -i "s/seed: [0-9]*/seed: $seed/" $yaml_file
  
  # Run the command
  $docker
  
  # Wait for the command to finish
  wait
done