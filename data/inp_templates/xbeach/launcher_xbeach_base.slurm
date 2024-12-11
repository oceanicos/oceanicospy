#!/bin/bash
#SBATCH --account=punim1660
#SBATCH --job-name="case_number"
#SBATCH --output=path_casecase_number.out
#SBATCH --partition=bigmem
#SBATCH --ntasks=32
#SBATCH --mem=800G
#SBATCH --time=20:00:00
#SBATCH --mail-user=franklin.ayalacruz@student.unimelb.edu.au
#SBATCH --mail-type=ALL

module purge
module load iompi netCDF-Fortran/4.6.0

# The job command(s):

cd path_case

export PATH=${PATH}:/home/fayalacruz/xbeach/bin/bin
export PATH=${PATH}:/home/fayalacruz/xbeach/bin/lib
srun --ntasks=32 xbeach 

##Job monitor command to list the resource usage
my-job-stats -a -n -s