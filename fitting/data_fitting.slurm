#!/bin/bash

### PYMDP DEPRESSION SLURM - THIS SCRIPT WILL ONLY WORK IF YOU COPY IT TO ONE OF YOUR OWN FOLDERS
### AND START IT FROM THERE!

### Scheduling Parameters
# Number of processes. The total number of processes should be less than
# number of subjects*number of repeats per subject
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# runtime, just less than a day
#SBATCH --time=23:59:00
# We only need 32GB of memory  in total
#SBATCH --mem=32G
# Job output will be written to this file. %A will be replaced by job id
#SBATCH --output=pymdp_depression_%A.out
# Job name
#SBATCH --job-name=pymdp_depression
# TAM people get priority
#SBATCH --partition=owner_tam
#SBATCH --mail-user=eckertan@staff.uni-marburg.de
#SBATCH --mail-type=END


### Environment Setup
# Start with an empty environment
module purge
# Load conda environment that we created for this job
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate scikit
export LC_ALL=C
export OMP_NUM_THREADS=1

### Program calls to start the calculation
# Any allowed commands for bash scripts / linux command line can be used here
# Then  we start the script
python3 -O data_fitting.py --numprocs=$SLURM_CPUS_PER_TASK
