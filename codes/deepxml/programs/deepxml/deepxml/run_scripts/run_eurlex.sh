#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=logs5.txt

source ../../../../../../../miniconda3/bin/activate xml
bash run_main.sh 0 DeepXML-DeepWalk EURLex-4K 0 108
