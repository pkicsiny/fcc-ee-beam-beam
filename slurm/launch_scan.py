# -*- coding: utf-8 -*-

import shutil,os
import numpy as np
import argparse
from datetime import datetime
import itertools

today = datetime.today().strftime("%Y_%m_%d")

# change the executable file name here
exec_key = "ws_fma"
python_exec = f"exec_{exec_key}.py"

# estimated simulation time
walltime = "2:00:00"  # [h:mm:ss] 1 day = 86400 s

# these can be arrays for parameter scans
# you can define new parameters here too
n_threads_arr    = np.array([8], dtype=int)

n_macroparts_arr = np.array([1e3], dtype=int)

n_turns_arr      = np.array([500], dtype=int)

n_slices_arr     = np.array([100], dtype=int)  # [1] for this study submit the set of jobs for each num slices in parallel

# change these to the desired input and output path
# the folder name is used as a prefix to the simulation name
study_name = os.getcwd().split("/")[-1]
afs_dir = f'/home/HPC/pkicsiny/fccee-beambeam/tutorials/{study_name}' # directory to store input files (needed for HTCondor which cannot read from EOS)
eos_dir = f'/home/HPC/pkicsiny/fccee-beambeam/tutorials/{study_name}/outputs' # directory to store the output data on eos at the end of the execution (to avoid filling the limited AFS space)

###################
# parameter scans #
###################

# parameters to scan should be added in this list
grid = itertools.product(n_threads_arr, n_macroparts_arr, n_turns_arr, n_slices_arr)

for th, nm, nt, ns in grid:
    print("[launch_scan.py] --------")

    # type conversions for safety
    th = int(th)
    nm = int(nm)
    nt = int(nt)
    ns = int(ns)

    ######################
    # create directories #
    ######################

    # create directories separately
    sim_key = "{}_{}_{}_th_{:.2e}_mp_{:.2e}_tn_{:.2e}_sl".format(today, exec_key, th, nm, nt, ns)

    # file name has to be less than 256 characters
    len_simkey = len(sim_key)
    print(f"[launch_scan.py] len simkey: {len_simkey}")
    if len_simkey > 242:  # 242 is still ok, len(12345678.0.out)=14, 256-14=242
        raise ValueError(f"[launch_scan.py] simkey too long: {sim_key}")

    # input dir contains jobs submission files, output dir contains plots or turn by turn data etc.
    input_dir = os.path.join(afs_dir+"/inputs",sim_key)
    output_dir = os.path.join(eos_dir,sim_key)

    print("[launch_scan.py] Creating {} with walltime {} [s]".format(sim_key, walltime))

    #clean up contents of input and output folders of the same name
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(input_dir)
    os.makedirs(output_dir)

    # create submission files
    exec_file_name = 'slurm_job.sh'
    exec_file = os.path.join(input_dir, exec_file_name)
    shutil.copyfile(os.path.join(afs_dir, 'template.sh'), exec_file)
    os.system("sed -i 's#%SIMKEY#"+sim_key+"#g' " + exec_file)   

    # submit job
#    command = f"sbatch -t {walltime} -n 1 -c 1 {exec_file} {th} {nm} {nt} {ns} {python_exec} {walltime} {output_dir}"  # GPU
    command = f"sbatch --partition=slurm_hpc_acc -t {walltime} -n 1 -c {th} {exec_file} {th} {nm} {nt} {ns} {python_exec} {walltime} {output_dir}"  # CPU

    print(f"[launch_scan.py] command: {command}")
    os.system(command)
