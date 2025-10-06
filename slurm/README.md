Example scripts to run on CERN's slurm cluster:

- __exec_ws.py__: similar to 002_linear_tracking_with_ws_beambeam.ipynb
- __exec_ss.py__: similar to 003_linear_tracking_with_ss_beambeam.ipynb
- __exec_ws_fma.py__: similar to 004_linear_tracking_with_ws_beambeam_fma.ipynb

__launch_scan.py__

launch script to submit jobs to the cluster
in here you can define:
- python executable from above
- input/output directories (output dir must match the dir defined at the top of __template.sh__)
- parameters to scan

__template.sh__

this is a bash script and it will be executed once the job is given
a compute node on the cluster. This script activates the python environment but you
need to specify the path to it. 
Replace `/home/HPC/pkicsiny/fccee-beambeam/tutorials/slurm` with the actual path
to these files.
Replace `/home/HPC/pkicsiny/miniforge3/bin/activate` with the actual path
to the activation script to your conda environment. You can find it with `which conda`.
Then it calls the python executable with a set of parameters. The bash script also creates the output 
directory specified in launch_scan.py (the eos_dir).

Launch the job submission by:
`python launch_scan.py`
this will submit the job to the cluster and call the template.sh bash script on it.
The bash script launches the exec.py simulation.
