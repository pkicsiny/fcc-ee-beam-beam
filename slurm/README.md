Example scripts to run on CERN's slurm cluster:

- __exec_ws.py__: similar to 002_linear_tracking_with_ws_beambeam.ipynb
- __exec_ss.py__: similar to 003_linear_tracking_with_ss_beambeam.ipynb
- __exec_ws_fma.py__: similar to 004_linear_tracking_with_ws_beambeam_fma.ipynb

__launch_scan.py__

Launch script to submit jobs to the cluster. What you need to define/modify:
- python executable from above
- input/output directories (`output_dir` must match the dir defined at the top of __template.sh__)
- parameters to scan

__template.sh__

This is a bash script and it will be executed once the job is allocated
a compute node on the cluster. The properties in the script are filled in by __launch_scan.py__
You have to replace:
- path to output and error files at the top: `/home/HPC/pkicsiny/fccee-beambeam/tutorials/slurm`. 
The path set must match `output_dir` in __launch_scan.py__.
- `/home/HPC/pkicsiny/miniforge3/bin/activate` with the conda activation script in your file system.
You can find this by typing `which conda`.
The bash script calls the python executable with one set of parameters.

Launch the job submission by:
`python launch_scan.py`
this will submit the job to the cluster and call the template.sh bash script on it.
The bash script launches the exec.py simulation.
