#!/bin/bash
#SBATCH -J diagn
#SBATCH -A torrey-group # Change to your allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -p standard
#SBATCH --array=1-10 # Add run numbers here
#SBATCH --output=logs/diagnose_%j.out 
#SBATCH --error=logs/diagnose_%j.err 
#SBATCH --mail-user=yja6qa@virginia.edu #  Change to your email
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs

module purge
module load miniforge  # Loading conda on Rivanna
conda activate kho_env # Change to your conda env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python run_all_diagnostics.py $SLURM_ARRAY_TASK_ID
