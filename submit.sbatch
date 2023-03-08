#!/bin/bash
#
#SBATCH --job-name=ruck_and_roll # Job name for tracking
#SBATCH --partition=falcon,gpu-batch   # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --output=slurms/slurm_%j.out # Standard out from your job

## Partitions can be `falcon` or `falcon-debug` for when testing to see if it works

source venv/bin/activate

source /etc/profile.d/modules.sh
module load CUDA

srun python -m game_label_model --u1903266
