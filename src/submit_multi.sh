#!/bin/bash

#SBATCH -J multi_dbm
#SBATCH -o ../outputs/dbm_multi_output_%j.txt
#SBATCH -e ../outputs/dbm_multi_error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH -q hopper
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH -A r00939

module load conda
module load cudatoolkit/12.6
conda activate jax_env

# Debug & reliability
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # prints DDP init details
export NCCL_DEBUG=INFO                    # NCCL layer logs (can switch to WARN later)
export CUDA_LAUNCH_BLOCKING=0             # set 1 if you suspect CUDA kernel errors

# Per-rank log dir for torchrun (you'll get rank_*/stderr,stdout)
LOGDIR=../outputs/torchrun_logs_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

srun torchrun \
  --standalone \
  --max_restarts=0 \
  --log_dir="$LOGDIR" \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  train_hpc.py \
    --distributed \
    --model_type multinomial \
    --learning_type unsupervised \
    --epoch 10000 \
    --data_path /N/slate/zwu1/CancerDrugCell/cell_drug_response_samples.pt

