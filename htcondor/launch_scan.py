# -*- coding: utf-8 -*-

import shutil,os
import numpy as np
import argparse
from datetime import datetime
import itertools

today = datetime.today().strftime("%Y_%m_%d")

# change the executable file name here
exec_key = "ss"
python_exec = f"exec_{exec_key}.py"

# estimated simulation time
walltime = 1*86400  # [s] 1 day = 86400 s

# these can be arrays for parameter scans
# you can define new parameters here too
n_threads_arr    = np.array([8], dtype=int)

n_macroparts_arr = np.array([1e5], dtype=int)

n_turns_arr      = np.array([500], dtype=int)

n_slices_arr     = np.array([100], dtype=int)  # [1] for this study submit the set of jobs for each num slices in parallel

# change these to the desired input and output path
# the folder name is used as a prefix to the simulation name
study_name = os.getcwd().split("/")[-1]
afs_dir = f'/afs/cern.ch/work/p/pkicsiny/private/git/fccee-beambeam/tutorials/{study_name}' # directory to store input files (needed for HTCondor which cannot read from EOS)
eos_dir = f'/eos/home-p/pkicsiny/share/{study_name}' # directory to store the output data on eos at the end of the execution (to avoid filling the limited AFS space)

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

    exec_file_name = 'test_ss.sh'
    job_file_name = 'test_ss.job'

    shutil.copyfile(os.path.join(afs_dir, 'template.sh'), os.path.join(input_dir,exec_file_name))
    shutil.copyfile(os.path.join(afs_dir, 'template.job'), os.path.join(input_dir,job_file_name))

    ###########################################################
    # replace the placeholder strings with the specific value #
    ###########################################################

    # .job/.sub (jobfile) specify requirements how many CPU/GPU to use
    if th >= 0:
        print("[launch_scan.py] Running on CPU...")
        os.system("sed -i 's#%CONTEXT#RequestCpus = max({1, %NTHREADS})#g' "           + os.path.join(input_dir,job_file_name))
    else:
        print("[launch_scan.py] Running on A100/V100 GPU...")
        os.system("sed -i 's/%CONTEXT/RequestCpus = 1\\\nRequestGpus = 1/g' "          + os.path.join(input_dir,job_file_name))

    os.system("sed -i 's#%SIMKEY#"+str(sim_key)+"#g' "                                 + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%EXECFILENAME#"+os.path.join(input_dir,exec_file_name)+"#g' " + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%WALLTIME#"+str(walltime)+"#g' "                              + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%INPUTDIR#"+input_dir+"#g' "                                  + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%NTHREADS#"+str(th)+"#g' "                                    + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%NMACROPARTS#"+str(nm)+"#g' "                                 + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%NTURNS#"+str(nt)+"#g' "                                      + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%NSLICES#"+str(ns)+"#g' "                                     + os.path.join(input_dir,job_file_name))
    os.system("sed -i 's#%PYTHONEXEC#"+str(python_exec)+"#g' "                         + os.path.join(input_dir,job_file_name))               

    #.sh (execfile)
    os.system("sed -i 's#%OUTDIR#"+output_dir+"#g' "                                   + os.path.join(input_dir,exec_file_name))

    ################################
    # finally submit job to condor #
    ################################

    # "put condor_submit -i " for interactive jobs
    os.system('condor_submit ' + os.path.join(input_dir,job_file_name))
