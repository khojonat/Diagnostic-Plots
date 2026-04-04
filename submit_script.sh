#!/bin/bash
#SBATCH -J diagn
#SBATCH -A torrey-group
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -p standard
#SBATCH --output=logs/diagnose_%j.out 
#SBATCH --error=logs/diagnose_%j.err 
#SBATCH --mail-user=yja6qa@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load miniforge
conda activate kho_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_all_diagnostics.py /project/torrey-group/jkho/Kho_FIRE_MW_Suite/run_0/zoom/RUNs/output/ 90 --box-num 3 
