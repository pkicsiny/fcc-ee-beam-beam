# author: Peter Kicsiny
# source /afs/cern.ch/work/p/pkicsiny/public/miniforge3/bin/activate base
# python3 ./exec_ws_fma.py --nthreads 0 --nmacroparts 1000 --nturns 1024  --nslices 100 --outdir .

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 32
matplotlib.rcParams['figure.subplot.left'] = 0.18
matplotlib.rcParams['figure.subplot.bottom'] = 0.16
matplotlib.rcParams['figure.subplot.right'] = 0.92
matplotlib.rcParams['figure.subplot.top'] = 0.9
matplotlib.rcParams['figure.figsize'] = (12,8)
import time
import argparse
import os
import scipy.constants as cst

import sys
sys.path.append("/home/HPC/pkicsiny/harpy") # change this path to the installation folder of harpy, https://gitlab.cern.ch/jcoellod/harpy.git
import harmonic_analysis as ha

parser = argparse.ArgumentParser()

print(xo.__version__, xt.__version__, xf.__version__, xp.__version__)

parser.add_argument("--nthreads")
parser.add_argument("--nmacroparts")
parser.add_argument("--nturns")
parser.add_argument("--nslices")
parser.add_argument("--outdir")

args = parser.parse_args()  # all are string

arg_n_threads    =         int(args.nthreads)  # int
arg_n_macroparts =         int(args.nmacroparts)  # int
arg_n_turns      =         int(args.nturns)  # int
arg_n_slices     =         int(args.nslices)  # int
arg_out_dir      =         str(args.outdir)  # string

###############
# output path #
###############

outputs_path = os.path.join(arg_out_dir, "outputs")
os.makedirs(outputs_path, exist_ok=True)

####################
# helper functions #
####################

def do_fft_dump(coords_buffer, qx, qy, qx_i_anal, qy_i_anal, n_macroparts, n_turns, fname_idx=0, window=3, laskar=True, laskar_n_peaks=2, laskar_lower=.9, laskar_upper=1.2, outputs_path="../outputs/fma", out_name=None, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    :param coords_buffer: dict or pandas dataframe. If df each column is a series of length n_turns. 
    If dict each value is an np array of shape (n_macroparticles, n_turns)
    :param qx, qy (float): nominal tunes in units of [2pi]. Full turn tune for nonlinear lattice, superperiod tune for linear tracking.
    :param qx_i_anal, qy_i_anal (float): analytical incoherent tunes in units of [2pi]. Full turn tune for nonlinear lattice, superperiod tune for linear tracking. If there is no beambeam in the simulation, use q_i_anal=q.
    """
        
    #normalize    
    if beta_x > 0:
        coords_buffer = normalize_phase_space(coords_buffer, alpha_x=alpha_x, alpha_y=alpha_y, beta_x=beta_x, beta_y=beta_y)
        key_x, key_y, key_px, key_py = "x_norm", "y_norm", "px_norm", "py_norm"
    else:
        key_x, key_y, key_px, key_py = infer_buffer_type(coords_buffer)      
        
    print("Computing tune spectra...")
    q_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim = do_fft(coords_buffer, n_macroparts, n_turns,
                                                                      qx, qy,
                                                                      qx_i_anal, qy_i_anal,
                                                                      window=window, 
                                                                      laskar=laskar,
                                                                      laskar_n_peaks=laskar_n_peaks,
                                                                      key_x=key_x, key_y=key_y, laskar_lower=laskar_lower, laskar_upper=laskar_upper)
    # write running tunes
    print(f"Saving incoherent tunes to {outputs_path}")
    if out_name is None:
        fname = os.path.join(outputs_path, "q_i_sim_{}.txt".format(np.char.zfill(str(fname_idx),3)))
    else:
        fname = os.path.join(outputs_path, out_name)
    np.savetxt(fname, np.c_[qx_i_sim, qy_i_sim], header="qx_i_sim qy_i_sim")

def do_fft(coords_dict, n_macroparts, n_turns, bare_tunes, incoherent_tunes, window=3, laskar=True, laskar_n_peaks=2, key_x="x", key_y="y", key_s="zeta", laskar_lower=.95, laskar_upper=1.2):
    """
    :param coords_dict (dict): contains particle trajectories. Keys are dynamical variables, values are np arrays of shape (n_macroparticles, n_turns).
    :param qx, qy (float): nominal fractional tunes in units of [2pi]. Typically the full turn tune with full lattice tracking and tune per superperiod for linear tracking.
    :param qx_i_anal, qy_i_anal (float): fractional incoherent tunes in units of [2pi]. Tune of a small amplitude particle. Typically the full turn tune with full lattice tracking and tune per superperiod for linear tracking.
    :param laskar (bool): find peaks with laskar method i.e. interpolation, see: [Application of frequency map analysis to the ALS, J. Laskar] (true) OR by selecting the largest amplitude FFT channel (false)
    :param window (int): if 'laskar'=False, search for the FFT peak in the neighborhood [q_i_anal_idx-window, q_i_anal_idx+window[, where 'q_i_anal_idx' is the FFT channel (array index) of the peak of the analytical incoherent tune. Accuracy of peak is 1/n_turns.
    :param laskar_n_peaks (int): if 'laskar'=True, select this many largest amplitude peaks from the spectrum. Usually even number as the FFT spectrum is symmetric to 0.5, so there are 2 copies of each peak. Accuracy of peak is 1/n_turns**4.
    :param laskar_lower, laskar_upper (float): from the 'laskar_n_peaks' peaks found by the laskar method, trigger on those within the window [q-lower, q_i_anal+upper], i.e. between the nominal and analytical beam-beam incoherent tunes.
    """    
    
    if len(bare_tunes) == 2 and len(incoherent_tunes) == 2:
        qx, qy = bare_tunes
        qx_i_anal, qy_i_anal = incoherent_tunes
        do_qs = False
    elif len(bare_tunes) == 3 and len(incoherent_tunes) == 3:
        qx, qy, qs = bare_tunes
        qx_i_anal, qy_i_anal, qs_i_anal = incoherent_tunes
        do_qs = True
    else:
        raise ValueError("tunes tuple length has to be either 2 or 3")
    
    coords_x = np.reshape(coords_dict[key_x], (n_macroparts, n_turns))
    coords_y = np.reshape(coords_dict[key_y], (n_macroparts, n_turns))
    if do_qs:
        coords_s = np.reshape(coords_dict[key_s], (n_macroparts, n_turns))
    
    length = n_macroparts  # number of macroparticles
    fft_resolution = n_turns  # fft resolution is equal to the number of time samples
    fft_x_single_part = np.zeros((length, fft_resolution))  # tune spectrum x
    fft_y_single_part = np.zeros((length, fft_resolution))  # tune spectrum y
    if do_qs:
        fft_s_single_part = np.zeros((length, fft_resolution))  # tune spectrum s

    qx_i_sim = np.zeros((length))  # incoherent tune peak x
    qy_i_sim = np.zeros((length))  # incoherent tune peak y
    if do_qs:
        qs_i_sim = np.zeros((length))  # incoherent tune peak s
    
    ##############
    # fft x axis #
    ##############
    
    if qx_i_anal>.5:
        qx_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))
    else:
        qx_rel = np.fft.fftfreq(fft_resolution)
        
    if qy_i_anal>.5:
        qy_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))
    else:
        qy_rel = np.fft.fftfreq(fft_resolution)

    if do_qs:
        if qs_i_anal>.5:
            qs_rel = np.fft.fftshift(np.fft.fftfreq(fft_resolution))
        else:
            qs_rel = np.fft.fftfreq(fft_resolution)

        
    # get spectrum of all particle trajectories
    for part_i in range(length):
        
        if qx_i_anal>.5:
            fft_x_single_part[part_i]  = np.log10(np.abs(np.fft.fft(coords_x[part_i])))
        else:
            fft_x_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_x[part_i]))))

        if qy_i_anal>.5:
            fft_y_single_part[part_i]  = np.log10(np.abs(np.fft.fft(coords_y[part_i])))
        else:
            fft_y_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_y[part_i]))))

        if do_qs:
            if qs_i_anal>.5:
                fft_s_single_part[part_i]  = np.log10(np.abs(np.fft.fft(coords_s[part_i])))
            else:
                fft_s_single_part[part_i]  = np.log10(np.abs(np.fft.fftshift(np.fft.fft(coords_s[part_i]))))
            
        
        ################################
        # find shifted incoherent tune #
        ################################
        
        if laskar:
            
            # better approximation with Laskar frequency analysis: https://link.springer.com/content/pdf/10.1007/BF00699731.pdf
            fft_harpy_x_single_part = ha.HarmonicAnalysis(coords_x[part_i])
            fft_harpy_y_single_part = ha.HarmonicAnalysis(coords_y[part_i])
            if do_qs:
                fft_harpy_s_single_part = ha.HarmonicAnalysis(coords_s[part_i])

            f_x_single_part, coeff_x_single_part = fft_harpy_x_single_part.laskar_method(laskar_n_peaks)
            f_y_single_part, coeff_y_single_part = fft_harpy_y_single_part.laskar_method(laskar_n_peaks)
            if do_qs:
                f_s_single_part, coeff_s_single_part = fft_harpy_s_single_part.laskar_method(laskar_n_peaks)

            if qx>=.5:
                coeff_x_single_part = np.array(coeff_x_single_part)[(np.array(f_x_single_part)>=.5)]
                f_x_single_part     = np.array(    f_x_single_part)[(np.array(f_x_single_part)>=.5)]
            else:
                coeff_x_single_part = np.array(coeff_x_single_part)[(np.array(f_x_single_part)<.5)]
                f_x_single_part     = np.array(    f_x_single_part)[(np.array(f_x_single_part)<.5)]
                
            if qy>=.5:
                coeff_y_single_part = np.array(coeff_y_single_part)[(np.array(f_y_single_part)>=.5)]
                f_y_single_part     = np.array(    f_y_single_part)[(np.array(f_y_single_part)>=.5)]
            else:
                coeff_y_single_part = np.array(coeff_y_single_part)[(np.array(f_y_single_part)<.5)]
                f_y_single_part     = np.array(    f_y_single_part)[(np.array(f_y_single_part)<.5)]

            if do_qs:
                if qs>=.5:
                    coeff_s_single_part = np.array(coeff_s_single_part)[(np.array(f_s_single_part)>=.5)]
                    f_s_single_part     = np.array(    f_s_single_part)[(np.array(f_s_single_part)>=.5)]
                else:
                    coeff_s_single_part = np.array(coeff_s_single_part)[(np.array(f_s_single_part)<.5)]
                    f_s_single_part     = np.array(    f_s_single_part)[(np.array(f_s_single_part)<.5)]
            
            coeff_x_single_part = np.array(coeff_x_single_part)[(np.array(f_x_single_part)>=qx*laskar_lower) & (np.array(f_x_single_part)<=qx_i_anal*laskar_upper)]
            coeff_y_single_part = np.array(coeff_y_single_part)[(np.array(f_y_single_part)>=qy*laskar_lower) & (np.array(f_y_single_part)<=qy_i_anal*laskar_upper)]
            if do_qs:
                coeff_s_single_part = np.array(coeff_s_single_part)[(np.array(f_s_single_part)>=qs*laskar_lower) & (np.array(f_s_single_part)<=qs_i_anal*laskar_upper)]

            f_x_single_part = np.array(f_x_single_part)[(np.array(f_x_single_part)>=qx*laskar_lower) & (np.array(f_x_single_part)<=qx_i_anal*laskar_upper)]
            f_y_single_part = np.array(f_y_single_part)[(np.array(f_y_single_part)>=qy*laskar_lower) & (np.array(f_y_single_part)<=qy_i_anal*laskar_upper)]
            if do_qs:
                f_s_single_part = np.array(f_s_single_part)[(np.array(f_s_single_part)>=qs*laskar_lower) & (np.array(f_s_single_part)<=qs_i_anal*laskar_upper)]
            
            #print(part_i, f_y_single_part, np.abs(coeff_y_single_part), "\n")

            ##############################################################################################
            # take peak that is closest to the analytical incoherent tune computed with formulas outside #
            ##############################################################################################
            
            if len(f_x_single_part)>0:
                qx_i_sim[part_i] = f_x_single_part[np.argmax(np.abs(coeff_x_single_part))]
                #qx_i_sim[part_i] = f_x_single_part[np.argmin(np.abs(np.array(f_x_single_part)-(qx_i_anal)))]
            else: 
                qx_i_sim[part_i] = 0
                
            if len(f_y_single_part)>0:
                qy_i_sim[part_i] = f_y_single_part[np.argmax(np.abs(coeff_y_single_part))]
                #qy_i_sim[part_i] = f_y_single_part[np.argmin(np.abs(np.array(f_y_single_part)-(qy_i_anal)))]
            else:
                qy_i_sim[part_i] = 0 

            if do_qs:
                if len(f_s_single_part)>0:
                    qs_i_sim[part_i] = f_s_single_part[np.argmax(np.abs(coeff_s_single_part))]
                    #qs_i_sim[part_i] = f_s_single_part[np.argmin(np.abs(np.array(f_s_single_part)-(qs_i_anal)))]
                else:
                    qs_i_sim[part_i] = 0 
        
        ###################
        # dont use laskar #
        ###################
        
        else:
            
            # conversion from tune value to fft channel idx
            qx_i_anal_idx_in_fft = (qx_i_anal*fft_resolution + fft_resolution/2)
            qy_i_anal_idx_in_fft = (qy_i_anal*fft_resolution + fft_resolution/2)
            if do_qs:
                qs_i_anal_idx_in_fft = (qs_i_anal*fft_resolution + fft_resolution/2)
        
            # take peak that is closest to the analytical incoherent tune computed with formulas outside
            qx_i_sim_idx_in_fft = int(qx_i_anal_idx_in_fft)-window + np.argmax(fft_x_single_part[part_i][int(qx_i_anal_idx_in_fft)-window:int(qx_i_anal_idx_in_fft)+window])
            qy_i_sim_idx_in_fft = int(qy_i_anal_idx_in_fft)-window + np.argmax(fft_y_single_part[part_i][int(qy_i_anal_idx_in_fft)-window:int(qy_i_anal_idx_in_fft)+window])
            if do_qs:
                qs_i_sim_idx_in_fft = int(qs_i_anal_idx_in_fft)-window + np.argmax(fft_s_single_part[part_i][int(qs_i_anal_idx_in_fft)-window:int(qs_i_anal_idx_in_fft)+window])

            # simulated incoherent tune from vanilla fft
            qx_i_sim[part_i] = qx_rel[qx_i_sim_idx_in_fft]
            qy_i_sim[part_i] = qy_rel[qy_i_sim_idx_in_fft]
            if do_qs:
                qs_i_sim[part_i] = qs_rel[qs_i_sim_idx_in_fft]
    
    if do_qs:
        return qx_rel, qy_rel, qs_rel, fft_x_single_part, fft_y_single_part, fft_s_single_part, qx_i_sim, qy_i_sim, qs_i_sim  # these arrays of length n_macroparts
    else:
        return qx_rel, qy_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim  # these arrays of length n_macroparts

def compute_dq_anal(beam_params, yokoya=1.3, m_0=cst.m_e, sigma_z_key="sigma_z_bs", sigma_y_key="sigma_y", beta_s_key="beta_s_bs"):

    tunes = {}
    
    # particle radius
    r0 = -beam_params["q_b1"]*beam_params["q_b2"]*cst.e**2/(4*np.pi*cst.epsilon_0*m_0*cst.c**2) # - if pp
    
    # geometric reduction factor, piwinski angle
    phi_x = np.arctan(np.tan(beam_params["phi"])*np.cos(beam_params["alpha"]))
    phi_y = np.arctan(np.tan(beam_params["phi"])*np.sin(beam_params["alpha"]))
    
    piwi_x = beam_params[sigma_z_key]/beam_params[  "sigma_x"]*np.tan(phi_x)
    piwi_y = beam_params[sigma_z_key]/beam_params[sigma_y_key]*np.tan(phi_y)
    
    geometric_factor_x = np.sqrt(1 + piwi_x**2)
    geometric_factor_y = np.sqrt(1 + piwi_y**2)
    
    # get exact ξ with formula, when far from res. it is the tune shift (to incoherent mode) for each parameter in parameter scan
    tunes["xi_x"] = beam_params["bunch_intensity"]*beam_params["beta_x"]*r0 / (2*np.pi*beam_params["gamma"]) / \
    (beam_params["sigma_x"]*geometric_factor_x* \
    (beam_params["sigma_x"]*geometric_factor_x + beam_params[sigma_y_key]*geometric_factor_y))
    
    tunes["xi_y"] = beam_params["bunch_intensity"]*beam_params["beta_y"]*r0 / (2*np.pi*beam_params["gamma"]) / \
    (beam_params[sigma_y_key]*geometric_factor_y* \
    (beam_params["sigma_x"]*geometric_factor_x + beam_params[sigma_y_key]*geometric_factor_y))
    
    tunes["xi_s"] = beam_params["bunch_intensity"]*beam_params[beta_s_key]*np.tan(phi_x)**2*r0 / (2*np.pi*beam_params["gamma"]) / \
    (beam_params["sigma_x"]*geometric_factor_x* \
    (beam_params["sigma_x"]*geometric_factor_x + beam_params[sigma_y_key]*geometric_factor_y))
    
    #print("xi_x: {}\nxi_y: {}".format(tunes["xi_x"], tunes["xi_y"]))
    
    # get analytical incoherent tune, plug in exact ξ from previous
    if beam_params["qx"]-int(beam_params["qx"]) <.5:
        tunes["qx_i_anal"] = (np.arccos(np.cos(2*np.pi*beam_params["qx"]) - 2*np.pi*tunes["xi_x"]*np.sin(2*np.pi*beam_params["qx"])))/(2*np.pi)
    else:
        tunes["qx_i_anal"] = 1 - (np.arccos(np.cos(2*np.pi*beam_params["qx"]) - 2*np.pi*tunes["xi_x"]*np.sin(2*np.pi*beam_params["qx"])))/(2*np.pi)
        
    if beam_params["qy"]-int(beam_params["qy"]) <.5:
        tunes["qy_i_anal"] = (np.arccos(np.cos(2*np.pi*beam_params["qy"]) - 2*np.pi*tunes["xi_y"]*np.sin(2*np.pi*beam_params["qy"])))/(2*np.pi)
    else:
        tunes["qy_i_anal"] = 1 - (np.arccos(np.cos(2*np.pi*beam_params["qy"]) - 2*np.pi*tunes["xi_y"]*np.sin(2*np.pi*beam_params["qy"])))/(2*np.pi)

    if beam_params["qs"]-int(beam_params["qs"]) <.5:
        tunes["qs_i_anal"] = (np.arccos(np.cos(2*np.pi*beam_params["qs"]) - 2*np.pi*tunes["xi_s"]*np.sin(2*np.pi*beam_params["qs"])))/(2*np.pi)
    else:
        tunes["qs_i_anal"] = 1 - (np.arccos(np.cos(2*np.pi*beam_params["qs"]) - 2*np.pi*tunes["xi_s"]*np.sin(2*np.pi*beam_params["qs"])))/(2*np.pi)
        
        
    # get analytical tune shift (equals ξ when far from resonance)
    tunes["dqx_anal"] = tunes["qx_i_anal"] - beam_params["qx"]
    tunes["dqy_anal"] = tunes["qy_i_anal"] - beam_params["qy"]
    tunes["dqs_anal"] = tunes["qs_i_anal"] - beam_params["qs"]
    
    # analytical pi mode corrected by yokoya
    tunes["qx_pi_anal"] = beam_params["qx"]+yokoya*tunes["dqx_anal"]
    tunes["qy_pi_anal"] = beam_params["qy"]+yokoya*tunes["dqy_anal"]
    tunes["qs_pi_anal"] = beam_params["qs"]+yokoya*tunes["dqs_anal"]
    
    return tunes

###################
# beam parameters #
###################

beam_params = {
"q_b1"                : -1,  # [e]
"q_b2"                : 1,  #  [e]
"bunch_intensity"     : 1.51e11,  # [1]
"energy"              : 45.6,  # [GeV]
"p0c"                 : 45.6*1e9,  # [eV]
"mass0"               : 511000,  # [eV]
"phi"                 : 15e-3,  # [rad] half crossing angle
"alpha"               : 0,  # [rad] crossing plane
"u_sr"                : 0.0394,  # [GeV]
"qx"                  : 0.54,  # [1] superperiod tune
"qy"                  : 0.55,  # [1]
"qs"                  : 0.0072,  # [1]
"physemit_x"          : 7.1e-10, # [m]
"physemit_y"          : 7.5e-13,  # [m]
"beta_x"              : 0.11,  # [m]
"beta_y"              : 0.0007,  # [m]
"sigma_z"             : 5.6e-3,  # [m] sr
"sigma_z_bs"          : 12.7e-3,  # [m] sr+bs
"sigma_delta"         : 3.9e-4,  # [1]
"sigma_delta_bs"      : 8.9e-4,  # [1] sr+bs
"n_ip"                : 4,  # [1]
"k2_factor" : 0.7,  # [1] crab-waist strength scaling factor
}
    
beam_params["sigma_x" ] = np.sqrt(beam_params["physemit_x"]*beam_params["beta_x"])  # [m]
beam_params["sigma_px"] = np.sqrt(beam_params["physemit_x"]/beam_params["beta_x"])  # [1]
beam_params["sigma_y" ] = np.sqrt(beam_params["physemit_y"]*beam_params["beta_y"])  # [m]
beam_params["sigma_py"] = np.sqrt(beam_params["physemit_y"]/beam_params["beta_y"])  # [1]
beam_params["beta_s"]        = beam_params["sigma_z"]/beam_params["sigma_delta"]  # [m]
beam_params["physemit_s"]    = beam_params["sigma_z"]*beam_params["sigma_delta"]  # [m]
beam_params["beta_s_bs"]     = beam_params["sigma_z_bs"]/beam_params["sigma_delta_bs"]  # [m]
beam_params["physemit_s_bs"] = beam_params["sigma_z_bs"]*beam_params["sigma_delta_bs"]  # [m]
beam_params["gamma"]         = beam_params[    "energy"] /(beam_params["mass0"]*1e-9)  # [1]

#####################
# define parrticles #
#####################

context = xo.ContextCpu(omp_num_threads=arg_n_threads)

n_j = int(np.sqrt(arg_n_macroparts)) # number of grid points in x and y
j_max = 3 # maximum extent in action space

# logarithmic spacing of grid points to have better resolution for small amplitudes
j_vec_x = np.linspace(0,j_max, n_j)**2/2
j_vec_y = np.linspace(0,j_max, n_j)**2/2

# smallest amplitude particle (if x,y=0, no interaction)
j_vec_x[0] = 1e-4
j_vec_y[0] = 1e-4

# convert into x,y space
x_test_vec = np.sqrt(2*j_vec_x)
y_test_vec = np.sqrt(2*j_vec_y)

# create mesh of particles in x and y
test_coords = [(x_test, y_test) for x_test in x_test_vec for y_test in y_test_vec]
x_arr = np.array([xys[0] for xys in test_coords])
y_arr = np.array([xys[1] for xys in test_coords])

# 0 values for other dynamical variables
empty_coord_vec = np.zeros(len(x_test_vec)*len(y_test_vec))

# create particle grid
test_grid = xp.Particles(
             _context = context,
            q0        = -1,
            p0c       = beam_params["p0c"],
            mass0     = beam_params["mass0"],
                     x= beam_params["sigma_x"]*x_arr,
                     y= beam_params["sigma_y"]*y_arr,
                  zeta=empty_coord_vec,
                    px=empty_coord_vec,
                    py=empty_coord_vec,
                 delta=empty_coord_vec,
                 weight=1)
n_macroparticles = test_grid._capacity

# save initial coordinates for amplitude space plot of fma
xx = test_grid.x/beam_params["sigma_x"]
yy = test_grid.y/beam_params["sigma_y"]

print(f"[exec.py] number of test particles in grid: {n_macroparticles} ({n_j} x {n_j})")

#########################
# define linear lattice #
#########################

beta_x_sext_left  = 3
beta_y_sext_left  = 500
beta_x_sext_right = 3
beta_y_sext_right = 500

alpha_x_sext_left  = 0
alpha_y_sext_left  = 0
alpha_x_sext_right = 0
alpha_y_sext_right = 0

# from IP to right crab sextupole
el_arc_left_b1 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beam_params["beta_x"], beta_x_sext_left],
    bety = [beam_params["beta_y"], beta_y_sext_left],
    alfx = [0, alpha_x_sext_left],
    alfy = [0, alpha_y_sext_left],
    bets = beam_params["beta_s"],
)

el_arc_mid_b1 = xt.LineSegmentMap(_context=context,
    qx =  beam_params["qx"], 
    qy =  beam_params["qy"] -.5 , # subtract .25*2 phase advance from small arcs
    qs =  beam_params["qs"],
    betx = [beta_x_sext_left, beta_x_sext_right],
    bety = [beta_y_sext_left, beta_y_sext_right],
    alfx = [alpha_x_sext_left, alpha_x_sext_right],
    alfy = [alpha_y_sext_left, alpha_y_sext_right],
    bets = beam_params["beta_s"],
)

# from left crab sextupole to IP2
el_arc_right_b1 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beta_x_sext_right, beam_params["beta_x"]],
    bety = [beta_y_sext_right, beam_params["beta_y"]],
    alfx = [alpha_x_sext_right, 0],
    alfy = [alpha_y_sext_right, 0],
    bets = beam_params["beta_s"],
)

# injection from initial distribution to right sextupole
el_inject_b1 = xt.LineSegmentMap(_context=context,
    betx = [beam_params["beta_x"], beta_x_sext_right],
    bety = [beam_params["beta_y"], beta_y_sext_right],
    alfx = [0, alpha_x_sext_right],
    alfy = [0, alpha_y_sext_right],
    bets = beam_params["beta_s"],
)

k2_left  = beam_params["k2_factor"] / (2 * beam_params["phi"] * beam_params["beta_y"] * beta_y_sext_left ) * np.sqrt(beam_params["beta_x"] / beta_x_sext_left )
k2_right = beam_params["k2_factor"] / (2 * beam_params["phi"] * beam_params["beta_y"] * beta_y_sext_right) * np.sqrt(beam_params["beta_x"] / beta_x_sext_right)

el_sextupole_left  = xt.Multipole(order=2, knl=[0, 0,   k2_left])
el_sextupole_right = xt.Multipole(order=2, knl=[0, 0, -k2_right])

slicer = xf.TempSlicer(_context=context, n_slices=arg_n_slices, sigma_z=beam_params["sigma_z_bs"], mode="unicharge")

#################################
# define the beam-beam elements #
#################################

el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,

        ########################################################################
        # this is the charge in units of elementary charges of the strong beam #
        ########################################################################

        other_beam_q0=1,

        ##################################################################
        # phi=crossing angle in radians, alpha=crossing plane in radians #
        ##################################################################

        phi=beam_params["phi"],
        alpha=0,

        ###############################################################################
        # slice intensity [num. real particles] n_slices inferred from length of this #
        ###############################################################################

        slices_other_beam_num_particles = slicer.bin_weights * beam_params["bunch_intensity"],

        ######################################
        # unboosted strong bunch RMS moments #
        ######################################

        slices_other_beam_Sigma_11    = arg_n_slices*[beam_params["sigma_x"]**2],
        slices_other_beam_Sigma_22    = arg_n_slices*[beam_params["sigma_px"]**2],
        slices_other_beam_Sigma_33    = arg_n_slices*[beam_params["sigma_y"]**2],
        slices_other_beam_Sigma_44    = arg_n_slices*[beam_params["sigma_py"]**2],

        # no x-y coupling now
        slices_other_beam_Sigma_12    = arg_n_slices*[0],
        slices_other_beam_Sigma_34    = arg_n_slices*[0],

        ###############################
        # only if beamstrahlung is on #
        ###############################

        # these can be hidden from the user in the future
        slices_other_beam_zeta_center = slicer.bin_centers,
        slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(beam_params["phi"]),  # boosted dz

)

# particles monitor
monitor_coords_b1 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=arg_n_turns, particle_id_range=(0,test_grid._capacity))

line = xt.Line(elements = [
                          monitor_coords_b1,
                          el_sextupole_right,
                          el_arc_right_b1,
                          el_beambeam_b1,
                          el_arc_left_b1,
                          el_sextupole_left,
                          el_arc_mid_b1,
])

line.build_tracker(_context=context)

################
# fft settings #
################

fma_chunk_size = int(arg_n_turns / 2)
fma_step_size = int(arg_n_turns*0.05)  # results in 1/(2*f)+1 points, here f=0.05
fma_counter = 1
chunk_id = 0

tunes = compute_dq_anal(beam_params, yokoya=1.3, m_0=cst.m_e, sigma_z_key="sigma_z_bs")

#########
# track #
#########

bare_tunes = (beam_params["qx"], beam_params["qy"])
incoherent_tunes = (tunes["qx_i_anal"], tunes["qy_i_anal"])

el_inject_b1.track(test_grid)
for turn in range(arg_n_turns):
    
    ################
    # track 1 turn #
    ################
    
    start_time = time.time()
    line.track(test_grid, num_turns=1)
    time_elapsed = time.time() - start_time

    ########################################################
    # compute partial tunes and save them at turn 9,19,... #
    ########################################################

    if fma_counter==fma_chunk_size:
        fma_counter -= fma_step_size
        chunk_id    = ((turn+1) - fma_chunk_size) / fma_step_size
        chunk_start = int(chunk_id*fma_step_size)
        chunk_end   = int(chunk_id*fma_step_size + fma_chunk_size)
        print(f"Turn {turn+1}: FMA chunk {chunk_id} | coords [{chunk_start}-{chunk_end}[")
        
        # extract transverse coordinates falling into the relevant sliding window from the monitor
        mon_data = monitor_coords_b1.to_dict()["data"]
        coords_dict = {}
        coords_dict["x"]  = np.reshape( mon_data["x"], (n_macroparticles, arg_n_turns))[:,chunk_start:chunk_end]
        coords_dict["y"]  = np.reshape( mon_data["y"], (n_macroparticles, arg_n_turns))[:,chunk_start:chunk_end]
        
        # call utility function
        qx_rel, qy_rel, fft_x_single_part, fft_y_single_part, qx_i_sim, qy_i_sim = do_fft(coords_dict, test_grid._capacity, 
                                                                      fma_chunk_size, bare_tunes, incoherent_tunes,
                                                                      laskar_n_peaks=4,
                                                                      )
        # write running tunes
        fname = os.path.join(outputs_path, f"q_i_sim_{str(int(chunk_id)).zfill(3)}.txt")
        print(f"Saving incoherent partial tunes to {fname}")
        np.savetxt(fname, np.c_[qx_i_sim, qy_i_sim], header="qx_i_sim qy_i_sim")
    

    fma_counter +=1

    print(f"Done tracking for 1 turn [{turn+1}/{arg_n_turns}]: {time_elapsed:.4f} [s]")

# fft full partilce trajectories
print("[exec.py] computing full trajectory FFT")
coords_dict = monitor_coords_b1.to_dict()["data"]
qx_rel, qy_rel, fft_x_single_part, fft_y_single_part, qx_sim, qy_sim = do_fft(coords_dict, test_grid._capacity, 
                                                                      arg_n_turns, bare_tunes, incoherent_tunes,
                                                                      laskar_n_peaks=4,
                                                                      )
fname_full = os.path.join(outputs_path, "q_sim.txt")
np.savetxt(fname_full, np.c_[qx_sim, qy_sim], header="qx_sim qy_sim")

print("[exec.py] successfully finished simulation")
