#!/bin/bash

#SBATCH -J dbn_ccl
#SBATCH -p hopper
#SBATCH -o ../outputs/output_%j.txt
#SBATCH -e ../outputs/error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH -A r00939
#SBATCH --gpus-per-node=1

#Load any modules that your program needs
module load conda/24.1.2
module load nvidia/21.5

conda activate hdp

#Run your program
srun python3 ./train_dbn.py 
# srun python3 ./dbm.py
# srun python3 ./hdp.py
# srun python3 ./hdp_dbm.py
