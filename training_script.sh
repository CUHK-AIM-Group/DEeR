#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH --partition=team1
#SBATCH -w node31
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

nvidia-smi

python main.py --exp_name='exp1' \
--data_dir='./data/Retinal_OCT-C8/' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.1 \
--local_model='CLIP' \
--dataset='OCT' \
--T=50 \
--E=5 \
--lr=0.01 \
--num_classes=8 \
--lora_r=8 \
--device=0 \
--method='LoRA' \
--is_DP=1 \
--C=0.1 \
--epsilon=0.1 \


python main.py --exp_name='exp1' \
--data_dir='/home/meiluzhu2/data/kvasir-dataset-v2-processed-224' \
--node_num=12 \
--iid=0 \
--dirichlet_alpha=0.1 \
--local_model='CLIP' \
--dataset='Kvasir' \
--T=50 \
--E=5 \
--lr=0.01 \
--num_classes=8 \
--lora_r=8 \
--device=0 \
--method='FedLoRA' \
--is_DP=1 \
--C=0.3 \
--epsilon=6 \

