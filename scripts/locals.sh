#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: bash run_experiments.sh <PATH_TO_DATASETS> <PATH_TO_RESULTS>"
  exit 1
fi

# Paths from arguments
export PATH_TO_DATASETS="$1"
export PATH_TO_RESULTS="$2"

# Datasets (keys of the Python dict)
datasets=(
  "UCF101"
  "dtd"
  "OxfordPets"
  "eurosat"
  "StanfordCars"
  "Caltech101"
  "Flower102"
  "fgvc_aircraft"
  "sun397"
  "Food101"
)

# Backbones
backbones=(
  "ViT-B/16"
  "ViT-L/14"
  "RN50"
  "RN101"
)

# Conformal methods
methods=(
  "topk"
  "lac"
  "aps"
  "raps"
)

# Nested loops
for backbone in "${backbones[@]}"; do
  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      echo "Dataset: $dataset | Backbone: $backbone | Method: $method"
      python main.py \
        --dataset "$dataset" \
        --backbone "$backbone" \
        --n_folds 10 \
        --conformal_method "$method" \
        --drop_soft_labels \
        --sim_transform sigmoid
    done
  done
done

