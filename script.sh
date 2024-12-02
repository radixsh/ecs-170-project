#!/bin/bash

# iterate over all possible combinations of 4, 8, 16, and 32 layers with
# node counts of 16, 32, 64, 128 in each layer
# examples:
# python train_multiple.py 128 64 32 16
# python train_multiple.py 64 32 16 128
# python train_multiple.py 16 32 64 128
# python train_multiple.py 128 128 64 128 32 16 16 32
# python train_multiple.py 16 16 16 32 64 128 64 32 16 32 128 16 16 32 64 128

# for number in 16 32 64 128
# do
#     for layer in 4 8 16 32
#     do
#         layers=""
#         for i in $(seq 1 $layer)
#         do
#             layers="${layers} ${number}"
#         done
# 
#         echo "Running: python train_multiple.py "
#         echo python train_multiple.py ${layers} # > "output_${number}_${layer}.txt" 2>&1
#     done
# done

# Define layers and node counts
layer_counts=(4 8 16 32)
node_counts=(16 32 64 128)

# Function to generate combinations of nodes for a given number of layers
generate_all_combinations() {
    local depth=$1
    local prefix=$2

    if [ "$depth" -eq 0 ]; then
        echo "$prefix"
    else
        for node in "${node_counts[@]}"; do
            generate_combinations $((depth - 1)) "$prefix $node"
        done
    fi
}

generate_combinations() {
    echo 32 16
    echo 64 32 16
    echo 16 32 64
    echo 128 64 32
    echo 128 64 32 16
    echo 256 128 64 32 16
    echo 512 256 128 64 32 16
}

# Generate all combinations of nodes for the given number of layers
combinations=$(generate_combinations)

# Execute the Python script for each combination
while read -r combination; do
    combination=$(echo $combination | xargs) # Trim any extra spaces
    cmd="python train_multiple.py $combination" 
    echo $cmd
    fn="outputs/output_$(echo $combination | tr ' ' '_')_new.txt"
    $cmd 2>&1 | tee $fn
done <<< "$combinations"
