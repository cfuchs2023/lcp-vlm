#!/bin/bash
#SBATCH --job-name=euvip_raps_sigmoid.job
#SBATCH --output=/auto/globalscratch/users/c/f/cfuchs/euvip_raps_sigmoid.%A_%a.out
#SBATCH --error=/auto/globalscratch/users/c/f/cfuchs/euvip_raps_sigmoid.%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000M
#SBATCH --mail-user=clement046fuchs@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1
#SBATCH --array=0-0

echo "Hello"
cd /home/users/c/f/cfuchs/github/cp_euvip/
export PATH_TO_DATASETS="/auto/globalscratch/users/c/f/cfuchs/CLIP_features/"
export PATH_TO_RESULTS="/auto/globalscratch/users/c/f/cfuchs/euvip_RAPS_sigmoid/"
export PATH_TO_FIGURES="/auto/globalscratch/users/c/f/cfuchs/euvip_RAPS_sigmoid/"
source /auto/home/users/c/f/cfuchs/virtualenvs/mm-env/bin/activate
# Datasets (keys of the Python dict)
datasets=(
  "UCF101"
  #"hmdb51"
  "dtd"
  "OxfordPets"
  "eurosat"
  "StanfordCars"
  #"Caltech101"
  "Flower102"
  "fgvc_aircraft"
  "sun397"
  "Food101"
  #"Kinetics400"
  # "imagenet"
  # "UCF101"
)

# Backbones (keys and values of the backbone dict)
backbones=(
  #"ViT-B/16"
  #"ViT-L/14"
  #"ViT-B/32"
  "RN50"
  "RN101"
)


# Nested loops
for key in "${backbones[@]}"; do
  for dataset in "${datasets[@]}"; do
    value="${backbones[$key]}"
    echo "Dataset: $dataset | Backbone key: $key | Backbone value: $value"
	
    #python localized_cp.py --dataset "$dataset" --backbone "$key" --n_folds 10 --conformal_method raps --drop_soft_labels
	#--grid_search
	python localized_cp.py --dataset "$dataset" --backbone "$key" --n_folds 10 --conformal_method raps --sim_transform sigmoid
  done
done

