#!/bin/bash

#SBATCH --partition=insa-gpu
#SBATCH --job-name=dynn-ab11
#SBATCH --output=out2/out.txt
#SBATCH --error=out2/err.txt
#SBATCH -w crn22
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30-00:00:00
#SBATCH --mem=40G


srun singularity run --nv --bind ./src/:/src/ --bind ./out2/:/out/ myimage01.sif /bin/bash -c "cd /src/ && python train_ab_11.py"
