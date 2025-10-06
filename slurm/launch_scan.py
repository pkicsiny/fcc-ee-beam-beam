# -*- coding: utf-8 -*-

import shutil,os
import numpy as np
import argparse
from datetime import datetime
import itertools

today = datetime.today().strftime("%Y_%m_%d")

# change the executable file name here
exec_key = "ws"
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
root_dir   = os.getcwd() # current dir
input_dir  = os.path.join(root_dir,  'inputs')
output_dir = os.path.join(root_dir, 'outputs') # directory to store the output of all simulations launched by this script

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
    sim_key = f"{today}_{exec_key}_{th}_th_{nm:.2e}_nm_{nt:.2e}_nt_{ns:.2e}_ns"

    # file name has to be less than 256 characters
    len_simkey = len(sim_key)
    print(f"[launch_scan.py] len simkey: {len_simkey}")
    if len_simkey > 242:  # 242 is still ok, len(12345678.0.out)=14, 256-14=242
        raise ValueError(f"[launch_scan.py] simkey too long: {sim_key}")

    # input dir contains jobs submission files created by the bash script, output dir contains plots or turn by turn data etc.
    sim_input_dir  = os.path.join( input_dir, sim_key)
    sim_output_dir = os.path.join(output_dir, sim_key)

    print(f"[launch_scan.py] Creating {sim_key} with walltime {walltime} [s]")

    #clean up contents of input and output folders of this simulation
    if os.path.exists(sim_input_dir):
        shutil.rmtree(sim_input_dir)
    if os.path.exists(sim_output_dir):
        shutil.rmtree(sim_output_dir)
    os.makedirs(sim_input_dir)
    os.makedirs(sim_output_dir)

    # create submission file from template.sh
    exec_file_name = 'slurm_job.sh'
    exec_file = os.path.join(sim_input_dir, exec_file_name)
    shutil.copyfile(os.path.join(root_dir, 'template.sh'), exec_file)
    os.system("sed -i 's#%SIMKEY#"+sim_key+"#g' " + exec_file)   

    # submit job
#    command = f"sbatch -t {walltime} -n 1 -c 1 {exec_file} {th} {nm} {nt} {ns} {python_exec} {walltime} {output_dir}"  # GPU
    command = f"sbatch --partition=slurm_hpc_acc -t {walltime} -n 1 -c {th} {exec_file} {th} {nm} {nt} {ns} {python_exec} {walltime} {output_dir}"  # CPU

    print(f"[launch_scan.py] command: {command}")
    os.system(command)
