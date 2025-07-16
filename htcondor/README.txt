Example scripts to run on CERN's HTCondor cluster
- exec_ws.py: similar to 002_linear_tracking_with_ws_beambeam.ipynb
- exec_ss.py: similar to 003_linear_tracking_with_ss_beambeam.ipynb
- exec_ws_fma.py: similar to 004_linear_tracking_with_ws_beambeam_fma.ipynb

- launch_scan.py: launch script to submit jobs to the cluster
in here you can define which python executable to use and
which parameters to scan. Inside this file you have to specify the paths to put the
input (called afs_dir) and output (called eos_dir) 
data. This you need to adapt to your own file system.

- template.job: template job submission file in HTCondor format. It has placeholder
strings which will be automatically replaced by the actual values from the launch_scan.py
upon launching the simulation. For each parameter configuration a copy of this template
will be created and placed in the input folder (defined in afs_dir in launch_scan.py)
Here at "transfer_input_files" you have to specify the python environment to be used
for the job on the cluster. For details on this you can see the CERN ticket with the
discussion of HTCondor load balancing:
https://cern.service-now.com/service-portal?id=ticket&table=incident&n=INC4351221

- template.sh: this is a bash script and it will be executed once the job is given
a compute node on the cluster. This script unpacks the zipped python environment,
activates it and calls the python executable with a set of parameters. The bash 
script also creates the output directory specified in launch_scan.py (the eos_dir).

Launch the job submission by:
python launch_scan.py
this will call the template.job submission file to submit the job to a compute node on the cluster
once submitted template.sh bash script will be executed which launches the exec.py simulation
