#!/bin/bash

#SBATCH --partition=insa-gpu
#SBATCH --job-name=dynn-model01
#SBATCH --output=/aschroed/dyNN-model01/out.txt
#SBATCH --error=/aschroed/dyNN-model01/err.txt
#SBATCH -w crn20
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

srun singularity run --nv --bind /calcul-crn20/aschroed/dyNN-model01/Datasets --bind /calcul-crn20/aschroed/dyNN-model01 --bind /calcul-crn20/aschroed/dyNN-model01/Checkpoints /calcul-crn20/aschroed/dyNN-model01/myimage01.sif /bin/bash -c "python main_train.py"
