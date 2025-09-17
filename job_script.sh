#!/bin/bash
#SBATCH -c 2                # Number of cores (-c)
#SBATCH -t 0-02:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=150           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts jobid

echo 'Begin'

module load python/3.10.9-fasrc01

source activate global_reward_ENV

python3 model.py
echo 'Done'
