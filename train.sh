#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='/home2/neha.sherin/FastSpeech2/batchtrain_out.txt'

module add u18/cuda/11.6
module add u18/cudnn/8.4.0-cuda-11.6

cd /ssd_scratch/cvit
[ -d neha ] || mkdir neha
cd neha
pwd
rsync -aP ada:/share3/neha.sherin/Blizzard/Blizzard_2/output .
rsync -aP ada:/share3/neha.sherin/Blizzard/Blizzard_2/preprocessed_data .

cd /home2/neha.sherin/FastSpeech2
pwd


#source miniconda3/bin
echo starting training
python train.py -p config/Blizzard/preprocess.yaml -m config/Blizzard/model.yaml -t config/Blizzard/train.yaml --restore_step 100000




