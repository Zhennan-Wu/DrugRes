#!/bin/bash

#SBATCH -J sm_dbm
#SBATCH -p hopper
#SBATCH -o ../outputs/dbm_super_multi_output_%j.txt
#SBATCH -e ../outputs/dbm_super_multi_error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH -A r00939

#Load any modules that your program needs
module load conda/24.1.2
module load nvidia/21.5

conda activate jax_env

#Run your program
srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$HOSTNAME:$((29500 + SLURM_JOB_ID % 1000)) \
  train_hpc.py --distributed --model_type multinomial --learning_type supervised

