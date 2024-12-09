#!/bin/bash
#SBATCH --account=punim1660
#SBATCH --job-name="case_number"
#SBATCH --output=path_case_gridcase_number.out
#SBATCH --partition=bigmem
#SBATCH --ntasks=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --mail-user=franklin.ayalacruz@student.unimelb.edu.au
#SBATCH --mail-type=ALL

module purge
module load GCC GCCcore CMake Ninja OpenMPI netCDF netCDF-Fortran/4.6.0

# The job command(s):

cd path_case_grid

# rm /data/projects/punim1660/runs_swan/smith_s2.65_d0.01/machinefile
MACHINEFILE="machinefile"
srun -l /bin/hostname | sort -n | awk '{print $2}' > $MACHINEFILE
export PATH=${PATH}:/data/projects/punim1660/swan_no_netcdf/bin
swanrun -input run.swn -mpi 32 > run.log
mkdir -p path_case_gridprint
mv run.prt-0* path_case_gridprint

# cp /home/fayalacruz/runs/swan_SAI/VL02_100/VL02_50_nest.NEST /home/fayalacruz/runs/swan_SAI/VL02_50/

# cd /home/fayalacruz/runs/swan_SAI/VL02_50
# srun -l /bin/hostname | sort -n | awk '{print $2}' > $MACHINEFILE
# swanrun -input VL02_50m.swn -mpi 32 > VL02_50m.log
# mkdir -p /home/fayalacruz/runs/swan_SAI/VL02_50/print
# mv swan_input_ppb.prt-0* /home/fayalacruz/runs/swan_SAI/VL02_50/print



##Job monitor command to list the resource usage
my-job-stats -a -n -s