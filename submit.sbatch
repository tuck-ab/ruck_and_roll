#!/bin/bash
#
#SBATCH --job-name=ruck_and_roll # Job name for tracking
#SBATCH --partition=dualgpu-batch   # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --time=0-24:00:00      # Job time limit set to 2 days (48 hours)

## Partitions can be `falcon` or `falcon-debug` for when testing to see if it works

source venv2/bin/activate

source /etc/profile.d/modules.sh

srun python -m game_label_model --model_num 5
