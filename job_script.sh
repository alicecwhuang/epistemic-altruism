#!/bin/bash
#SBATCH --account=def-aliceh
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=altruism
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=alice.huang@uwo.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

echo "Starting SLURM task ${SLURM_ARRAY_TASK_ID}..."

module load python/3.13
module load scipy-stack
source ~/.virtualenvs/ENV/bin/activate

# Pass the array task ID as argument to the script
python3 model.py ${SLURM_ARRAY_TASK_ID}

echo "Done with task ${SLURM_ARRAY_TASK_ID}"
