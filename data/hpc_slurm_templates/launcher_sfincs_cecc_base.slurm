#!/bin/bash
#SBATCH --job-name="$case_name"
#SBATCH --clusters=fisica
#SBATCH --output=$output_path_case%j.out
#SBATCH --error=$output_path_case%j.err
#SBATCH --partition=cpu.cecc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$number_tasks
#SBATCH --mem=32G
#SBATCH --time=48:00:00  # up to 4 days
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

unset  SINGULARITY_BINDPATH
export SINGULARITY_BINDPATH="/scratchsan/medellin/:/scratchsan/medellin/"

> $run_path_case/log.txt

START=$(date +%s)
echo "INICIO, $HOSTNAME $(date)" >> $run_path_case/log.txt 2>&1

cd $run_path_case
singularity exec /localapps/$image_name sfincs 2>&1 >> $run_path_case/log.txt

END=$(date +%s)
echo "FIN, $HOSTNAME $(date)" >> $run_path_case/log.txt 2>&1

# ----- compute formatted elapsed time -----
ELAPSED=$((END - START))

HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

printf "ELAPSED, %02d:%02d:%02d\n" "$HOURS" "$MINUTES" "$SECONDS" >> $run_path_case/log.txt
