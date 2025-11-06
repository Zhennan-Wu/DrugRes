#!/bin/bash
#SBATCH -J ddp_test
#SBATCH -p hopper
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:05:00
#SBATCH -A r00939
#SBATCH -o ddp_test_%j.out
#SBATCH -e ddp_test_%j.err

module load conda
conda activate jax_env

srun torchrun --standalone --nproc_per_node=2 -m torch.distributed.run --no_python -- echo "Hello from rank"
