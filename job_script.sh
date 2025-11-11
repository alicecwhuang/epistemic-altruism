#!/bin/bash
#SBATCH --account=def-aliceh
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=altruism
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=alice.huang@uwo.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

echo "Begin"

module load python/3.13
module load scipy-stack
source ~/.virtualenvs/ENV/bin/activate
python3 model.py

echo "Done"
