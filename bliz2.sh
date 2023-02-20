#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --output='/home2/neha.sherin/bliz_long/bliz2_out.txt'

module add u18/cuda/11.6
module add u18/cudnn/8.4.0-cuda-11.6

cd /ssd_scratch/cvit
[ -d neha ] || mkdir neha
cd neha
pwd
rsync -aP ada:/share3/neha.sherin/Blizzard/Blizzard_2/output .
rsync -aP ada:/share3/neha.sherin/Blizzard/Blizzard0/Blizzard .
[ -d preprocessed_data ] || mkdir preprocessed_data
cd preprocessed_data
pwd
rsync -aP ada:/share3/neha.sherin/Blizzard/Blizzard0/preprocessed_data/Blizzard1/TextGrid .

cd /home2/neha.sherin/bliz_long
pwd

echo starting preproc
python preprocess.py config/Blizzard/preprocess.yaml
echo done preproc

echo starting sorting train txt
python sortdata.py
echo done sorting

echo starting training
python train.py -p config/Blizzard/preprocess.yaml -m config/Blizzard/model.yaml -t config/Blizzard/train.yaml 
echo training complete




