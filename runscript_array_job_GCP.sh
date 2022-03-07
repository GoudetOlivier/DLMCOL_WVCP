#!/bin/bash

#SBATCH --array=2-5
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --mem 100000M 
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out
#SBATCH  -p A40-short

srun python main.py GCP DSJC500.5 --seed $SLURM_ARRAY_TASK_ID --k 47
