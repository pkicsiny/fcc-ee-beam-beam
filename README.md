formerly: https://gitlab.cern.ch/acc-models/fcc/fcc-ee-beam-beam

remote: https://gitlab.cern.ch/pkicsiny/cluster

Content: submission scripts for slurm like clusters such as on `bastion.cnaf.infn.it`. 


# Login to cluster

`ssh username@bastion.cnaf.infn.it`

`ssh ui-hpc2.cr.cnaf.infn.it`

# Submission scripts

The general directory tree looks like:

```
.
+-- study_name
|   +-- inputs
|       +-- simkey
|           +-- slurm-1234.err
|           +-- slurm-1234.out
|           +-- slurm_job.sh
|   +-- results
|       +-- simkey
|           +-- outputs
|           +-- plots
|   +-- template.sh
|   +-- launch_scan.py
|   +-- exec.py
```

`study_name` is the name of the folder given by the user. `simkey` is the name of an individual simulation and the string contains all relevant parameters that helps to identify it. Its length is constrained to be less than 256 characters long which is known to cause problems when submitted to HTCondor. On slurm it is not a problem if it is longer. The job submission script `slurm_job.sh` is specialized using a template file `template.sh`. 

In `launch_scan.py` one can define parameter scans with numpy arrays. The first half of the script is the definition of a set of numpy arrays for the various beam and simulation parameters. The second half of the script is a for loop iterating over all possible combinations of the parameter arrays, and for each one it creates a simulation job with the current `simkey` which is a concatenation of the current parameter values, for example: 

`simkey=2025_01_01_bhabha_nonlinear_tracking_exec_ws_ct_1.00e+00_ws_fcc_z_ac_ebe_tk_-9.99e+02_ey_32_th_2.00e+03_mp_2.00e+04_tn_-9.99e+02_sl_-9.99e+02_nb_1.50e-02_fi_2_sr_0_bs_0_bh_1.00e-04_cx_-9.99e+02_dy_2_bn_49`

For each job the above directory tree is created within the `study_name` folder. The submission script and a standard output and standard error file are put in the `inputs` folder and the simulation outputs will be put in the `results` folder. At the end of each iteration loop the template submission file is specialized by pasting the specific parameters into the placeholders. The specific `slurm_job.sh` now calls the python executable `exec.py` with the simulation parameters of the iteration loop. The simulation is run on the cluster by first activating the virtual environment and then claling `exec.py`.

After setting the numpy arrays with the parameters, the parameter scan is launched by typing 

`python launch_scan.py`

in the terminal.

The available computing resources on the slurm cluster can be printed with:

`cat /etc/slurm/slurm.conf`

This will give a list of compute nodes with the number of sockets (CPUs) and cores per socket. In the cluster partition called "CERN" there are 28 nodes, 2 CPU sockets per node, 24 cores per CPU, all cores can be hyperthreaded (typically not useful): 48 threads / 96 hyperthreads per node. 48*28 = 1344 single thread jobs possible at the same time.

References:
[1] https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html
[2] https://slurm.schedmd.com/slurm.conf.html
