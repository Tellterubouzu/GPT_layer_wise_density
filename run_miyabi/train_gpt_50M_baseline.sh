#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=2:00:00 
#PBS -W group_list=gp36
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -N re_gpt50base

#set -x  # 実行トレース
#echo "Shell flags: $-"
module purge
module load cuda/12.8
module load cudnn/9.10.1.4

export CC=gcc
export CXX=g++
export CUDA_VISIBLE_DEVICES=0

cd ${PBS_O_WORKDIR}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/train.py --config configs/scaling/train_gpt2_50m_baseline.json
python scripts/eval.py \
  --config configs/eval_fineweb.json \
  --run_dir runs/gpt2_50m_baseline/ \
  --output_file results/gpt2_50m_baseline_fineweb.json