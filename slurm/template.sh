#!/bin/bash

# change these paths to your file system
#SBATCH -e /home/HPC/pkicsiny/fccee-beambeam/tutorials/slurm/inputs/%SIMKEY/slurm-%j.err
#SBATCH -o /home/HPC/pkicsiny/fccee-beambeam/tutorials/slurm/inputs/%SIMKEY/slurm-%j.out

# uncomment this if you submit the job on a GPU
##SBATCH --gres=gpu:1

####################################
#assign the command line arguments #
####################################

# argument $9 but ${10}
nthreads=$1
nmacroparts=$2
nturns=$3
nslices=$4
pythonexec=$5
walltime=$6
outdir=$7
mkdir -p ${outdir}

#######################
# activate python env #
#######################

echo "[shell] Activating miniconda python libraries"
# replace this with your miniforge path
command="source /home/HPC/pkicsiny/miniforge3/bin/activate base"
echo '[shell] Command: ' ${command}
${command}

#############################
# set number of CPU threads #
#############################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# this prints the GPU details if there is a GPU on the node
command="nvidia-smi --query-gpu=name --format=csv,noheader"
${command}

######################
# execute simulation #
######################

echo "[shell] Executing simulation with walltime " ${walltime} " [s]"
command="srun python ./${pythonexec} \
            --nthreads ${nthreads} \
            --nmacroparts ${nmacroparts} \
            --nturns ${nturns} \
            --nslices ${nslices} \
            --outdir ${outdir}"
echo '[shell] Command: ' ${command}
${command}

######################
# print runtime info #
######################

command="sacct -j $SLURM_JOBID --format=jobid,jobname,elapsed,partition,ncpus,ntasks,state,exitcode"
echo "[shell] Requesting elapsed time: " ${command}
${command}

echo "[shell] Successfully finished xsuite job."
