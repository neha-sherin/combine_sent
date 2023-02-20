#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:1
#SBATCH -w gnode072
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

module add u18/cuda/11.6
module add u18/cudnn/8.4.0-cuda-11.6

#source miniconda3/bin

python synthesize.py --text "i am blizzard. i can understand full stops" --restore_step 100000 --mode single -p config/Blizzard/preprocess.yaml -m config/Blizzard/model.yaml -t config/Blizzard/train.yaml
