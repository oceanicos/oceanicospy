#!/bin/bash
#SBATCH --job-name="$case_name"
#SBATCH --cluster=fisica
#SBATCH --output=$run_path_case%j.out
#SBATCH --error=$run_path_case%j.err
#SBATCH --partition=cpu.cecc
#SBATCH --ntasks=$number_tasks
#SBATCH --mem=20G
#SBATCH --time=48:00:00  # up to 4 days
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

export I_MPI_HYDRA_BOOTSTRAP=ssh
unset  SINGULARITY_BINDPATH
export SINGULARITY_BINDPATH="/scratchsan/medellin/:/scratchsan/medellin/"

> $run_path_case/log.txt

START=$(date +%s)
echo "INICIO, $HOSTNAME $(date)" >> $run_path_case/log.txt 2>&1

singularity exec /localapps/$image_name mpirun -np $SLURM_NTASKS xbeach  2>&1 >> $run_path_case/log.txt
mv $run_path_case/*.nc $output_path_case

END=$(date +%s)
echo "FIN, $HOSTNAME $(date)" >> $run_path_case/log.txt 2>&1

# ----- compute formatted elapsed time -----
ELAPSED=$((END - START))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

printf "ELAPSED, %02d:%02d:%02d\n" "$HOURS" "$MINUTES" "$SECONDS" >> $run_path_case/log.txt
