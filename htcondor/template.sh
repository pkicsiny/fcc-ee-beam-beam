#!/bin/bash

####################################
#assign the command line arguments #
####################################

jobid=$1
nthreads=$2
nmacroparts=$3
nturns=$4
nslices=$5
pythonexec=$6

#######################################################
# specify output dir and create it if it doesnt exist #
#######################################################

outdir=%OUTDIR
mkdir -p ${outdir}

#####################
# unpack python env #
#####################

echo "[shell] Activating miniconda python libraries"
command="mkdir my_env"
${command}
command="tar -xzf miniforge3_xtrack_d.tar.gz -C my_env"
${command}
command="source my_env/bin/activate"
${command}
echo "[shell] using Python: $(which python)"

######################
# execute simulation #
######################

echo "[shell] Executing simulation with walltime " ${walltime} " [s]"
command="python ./${pythonexec} \
            --nthreads ${nthreads} \
            --nmacroparts ${nmacroparts} \
            --nturns ${nturns} \
            --nslices ${nslices} \
            --outdir ${outdir}"
echo '[shell] Command: ' ${command}
${command}

echo "[shell] Successfully finished xsuite job."
