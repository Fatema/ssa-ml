#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p "ug-gpu-small"
#SBATCH --qos="short"
#SBATCH -t 2-0

source /etc/profile

module load cuda/9.0

source ~/.venv/pt/bin/activate

python3 land_classifier.py
