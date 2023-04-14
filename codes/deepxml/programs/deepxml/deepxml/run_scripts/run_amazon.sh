#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END
#SBATCH --output=logs3.txt

source ../../../../../../../miniconda3/bin/activate xml
bash run_main.sh 0 DeepXML-DeepWalk Amazon-670K 0 108
