#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=logs1.txt

bash run_main.sh 0 DeepXML-DeepWalk AmazonTitles-670K 0 108
