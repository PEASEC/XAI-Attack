#!/bin/bash
#SBATCH -J XAIATTACK-QQP
#SBATCH --mail-type=END
#SBATCH -e ../logs/cluster/%x.err.%A_%a
#SBATCH -o ../logs/cluster/%x.out.%A_%a
#SBATCH --mem-per-cpu=64000
#SBATCH -t 40:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1

#---------------------------------------------------------------

module purge
module load gcc
module load python
module load cuda
module load cuDNN

nvidia-smi

cd ..

export DATASET="qqp"
export MODEL="distilbert-base-uncased"

srun python src/main.py --dataset $DATASET --model $MODEL --wandb_logging
srun python src/adversarial_testing.py --dataset $DATASET --model $MODEL --wandb_logging
srun python src/adversarial_transfer.py --dataset $DATASET --basemodel $MODEL --transfermodel distilbert-base-uncased --wandb_logging
srun python src/adversarial_transfer.py --dataset $DATASET --basemodel $MODEL --transfermodel bert-base-uncased --wandb_logging
srun python src/adversarial_transfer.py --dataset $DATASET --basemodel $MODEL --transfermodel roberta-base --wandb_logging
