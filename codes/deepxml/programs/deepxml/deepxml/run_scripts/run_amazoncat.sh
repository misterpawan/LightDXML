#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=logs2.txt

#conda init bash
#conda activate xml
source ../../../../../../../miniconda3/bin/activate xml

bash run_main.sh 0 DeepXML-DeepWalk AmazonCat-13K 0 108
