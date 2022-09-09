#!/bin/bash

#SBATCH --partition=insa-gpu
#SBATCH --job-name=dynn-model01
#SBATCH --output=out/out.txt
#SBATCH --error=out/err.txt
#SBATCH -w crn20
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30-00:00:00
#SBATCH --mem=4G


srun singularity run --nv --bind ./src/:/src/ --bind ./out/:/out/ myimage01.sif /bin/bash -c "cd /src/ && python main_train.py"
