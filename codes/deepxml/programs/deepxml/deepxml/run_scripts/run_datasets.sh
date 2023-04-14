#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=logs.txt

#conda activate XMLCNN
#bash run_main.sh 0 DeepXML EURLex-4K 0 108
bash run_main.sh 0 DeepXML-DeepWalk Wiki10-31K 0 108
#bash run_main.sh 0 DeepXML AmazonCat-13K 0 108
#bash run_main.sh 0 DeepXML AmazonTitles-670K 0 108
#bash run_main.sh 0 DeepXML WikiSeeAlsoTitles-350K 0 108
