#!/bin/bash
#SBATCH --job-name="simulation_name"
#SBATCH --cluster=fisica,cecc # puede ser cecc
#SBATCH --output=path_caseoutput/simulation_name_swan_%j.out
#SBATCH --error=path_caseoutput/simulation_name_swan_%j.err
#SBATCH --partition=cpu.cecc
#SBATCH --ntasks=64
#SBATCH --mem=40G
#SBATCH --time=20:00:00  # up to 4 days
#SBATCH --mail-user=ffayalac@unal.edu.co
#SBATCH --mail-type=ALL
#SBATCH --export=SCRATCH_DIR=/scratch/$SLURM_JOB_ACCOUNT/$SLURM_JOB_USER/$SLURM_JOB_ID

unset  SINGULARITY_BINDPATH
export SINGULARITY_BINDPATH="/scratchsan/medellin/:/scratchsan/medellin/"

cd $SCRATCH_DIR  

for domain in {1..number_domains}
do
cd "domain_0$domain"
> path_caseoutput/log_dom_0$domain.txt
echo "INICIO," $HOSTNAME `date`  2>&1 >>path_caseoutput/log_dom_0$domain.txt

if [ "$domain" -gt 1 ]; then
    prev_domain=$((domain - 1))
    mv "../domain_0$prev_domain/child_0$prev_domain.NEST" "child_0$prev_domain.NEST" 
fi
singularity run /localapps/swan_41.51.sif  swanrun -input run.swn -omp $SLURM_NTASKS   2>&1 >>path_caseoutput/log_dom_0$domain.txt
echo "FIN," $HOSTNAME `date`  2>&1 >>path_caseoutput/log_dom_0$domain.txt
cd ..
done
