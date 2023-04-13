#!/bin/bash
#
#SBATCH --job-name=ruck_and_roll # Job name for tracking
#SBATCH --partition=falcon     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --time=0-24:00:00      # Job time limit set to 2 days (48 hours)

## Partitions can be `falcon` or `falcon-debug` for when testing to see if it works

source venv/bin/activate

source /etc/profile.d/modules.sh

srun python game_label_model/graph_model/generation/graph_model_script.py --number 33
srun python game_label_model/graph_model/generation/graph_model_script.py --number 32
