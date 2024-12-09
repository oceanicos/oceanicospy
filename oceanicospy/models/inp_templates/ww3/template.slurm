#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Mon Dec 05 2022 15:12:24 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=bigmem

# The name of the job:
#SBATCH --job-name="name_job"

# The project ID which this job should run under:
#SBATCH --account="punim1660"

#SBATCH --nodes=2

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=70

# Use this email address:
#SBATCH --mail-user=franklin.ayalacruz@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The amount of memory in megabytes per node:
#SBATCH --mem=750GB

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-16:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load foss/2019b
module load netcdf/4.7.1

# The job command(s):
srun ./ww3_shel

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
