# author: Peter Kicsiny
# source /afs/cern.ch/work/p/pkicsiny/public/miniforge3/bin/activate base
# python3 ./exec_ss.py --nthreads 0 --nmacroparts 100000 --nturns 500  --nslices 100 --outdir .

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

###################
# beam parameters #
###################

bunch_intensity     = 1.55e11  # [1]
energy              = 182.5  # [GeV]
p0c                 = energy*1e9  # [eV]
mass0               = 511000  # [eV]
phi                 = 15e-3  # [rad] half xing
u_sr                = 10.42  # [GeV]
qx                  = 0.537  # [1] superperiod tune
qy                  = 0.546  # [1]
qs                  = 0.02275  # [1]
physemit_x          = 1.59e-09  # [m]
physemit_y          = 9e-13  # [m]
beta_x              = 1  # [m]
beta_y              = 1.6e-3  # [m]
sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
sigma_px            = np.sqrt(physemit_x/beta_x)  # [1]
sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
sigma_py            = np.sqrt(physemit_y/beta_y)  # [1]
sigma_z             = 1.81e-3  # [m] sr
sigma_z_bs          = 2.17e-3  # [m] sr+bs
sigma_delta         = 16e-4  # [1]
sigma_delta_bs      = 19.2e-4  # [1] sr+bs
beta_s              = sigma_z/sigma_delta  # [m]
physemit_s          = sigma_z*sigma_delta  # [m]
physemit_s_bs       = sigma_z_bs*sigma_delta_bs  # [m]
n_ip                = 4  # [1]
k2_factor = 0.4  # [1] crab-waist strength scaling factor
phi = 15e-3  # [rad] half crossing angle

#####################
# define parrticles #
#####################

n_macroparticles_b1 = arg_n_macroparts
n_macroparticles_b2 = n_macroparticles_b1

context = xo.ContextCpu(omp_num_threads=arg_n_threads)

# initial beam sizes
(sigma_x_init, sigma_y_init, sigma_z_init,
 sigma_px_init, sigma_py_init, sigma_delta_init,
 physemit_x_init, physemit_y_init, physemit_s_init) = (     sigma_x,      sigma_y,      sigma_z,
                                                           sigma_px,     sigma_py,  sigma_delta,
                                                       physemit_x, physemit_y, physemit_s
                                                      )
# synrad equilibrium beam sizes
(sigma_x_eq, sigma_y_eq, sigma_z_eq,
 sigma_px_eq, sigma_py_eq, sigma_delta_eq,
 physemit_x_eq, physemit_y_eq, physemit_s_eq) = (   sigma_x,    sigma_y,     sigma_z,
                                                   sigma_px,   sigma_py, sigma_delta,
                                                 physemit_x, physemit_y,  physemit_s
                                                )

(sigma_z_eq_bs, sigma_delta_eq_bs, physemit_s_eq_bs) = (sigma_z_bs, sigma_delta_bs, physemit_s_bs)

# for reproducibility fix the random seed
np.random.seed(2)

# this is an electron beam
particles_b1 = xp.Particles(
            _context = context,
            q0        = -1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x_init        *np.random.randn(n_macroparticles_b1),
            y         = sigma_y_init        *np.random.randn(n_macroparticles_b1),
            zeta      = sigma_z_init        *np.random.randn(n_macroparticles_b1),
            px        = sigma_px_init       *np.random.randn(n_macroparticles_b1),
            py        = sigma_py_init       *np.random.randn(n_macroparticles_b1),
            delta     = sigma_delta_init    *np.random.randn(n_macroparticles_b1),
            weight=bunch_intensity/n_macroparticles_b1
)

# this is a positron beam
particles_b2 = xp.Particles(
            _context = context,
            q0        = 1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x_init    *np.random.randn(n_macroparticles_b2),
            y         = sigma_y_init    *np.random.randn(n_macroparticles_b2),
            zeta      = sigma_z_init    *np.random.randn(n_macroparticles_b2),
            px        = sigma_px_init   *np.random.randn(n_macroparticles_b2),
            py        = sigma_py_init   *np.random.randn(n_macroparticles_b2),
            delta     = sigma_delta_init*np.random.randn(n_macroparticles_b2),
            weight=bunch_intensity/n_macroparticles_b2
            )

#########################
# define linear lattice #
#########################

emit_damping_rate_s  = 2 * u_sr / energy / n_ip
emit_damping_rate_x = emit_damping_rate_s/2
emit_damping_rate_y = emit_damping_rate_s/2
sigma_damping_rate_s =  emit_damping_rate_s / 2
sigma_damping_rate_x = sigma_damping_rate_s / 2
sigma_damping_rate_y = sigma_damping_rate_s / 2

beta_x_sext_left  = 3
beta_y_sext_left  = 500
beta_x_sext_right = 3
beta_y_sext_right = 500

alpha_x_sext_left  = 0
alpha_y_sext_left  = 0
alpha_x_sext_right = 0
alpha_y_sext_right = 0

# from IP to right crab sextupole (sy2r.2)
el_arc_left_b1 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beta_x, beta_x_sext_left],
    bety = [beta_y, beta_y_sext_left],
    alfx = [0, alpha_x_sext_left],
    alfy = [0, alpha_y_sext_left],
    bets = beta_s,
)

el_arc_mid_b1 = xt.LineSegmentMap(_context=context,
    qx =  qx,
    qy =  qy -.5 , # subtract .25*2 phase advance from small arcs
    qs =  qs,
    betx = [beta_x_sext_left, beta_x_sext_right],
    bety = [beta_y_sext_left, beta_y_sext_right],
    alfx = [alpha_x_sext_left, alpha_x_sext_right],
    alfy = [alpha_y_sext_left, alpha_y_sext_right],
    bets = beta_s,
    damping_rate_x     = sigma_damping_rate_x,
    damping_rate_px    = sigma_damping_rate_x,
    damping_rate_y     = sigma_damping_rate_y,
    damping_rate_py    = sigma_damping_rate_y,
    damping_rate_zeta  = sigma_damping_rate_s,
    damping_rate_pzeta = sigma_damping_rate_s,

    # noise has to be normalized to beamsize at exit using beta at exit!
    gauss_noise_ampl_x     = np.sqrt(physemit_x_eq*beta_x_sext_right) * np.sqrt(emit_damping_rate_x),
    gauss_noise_ampl_px    = np.sqrt(physemit_x_eq/beta_x_sext_right) * np.sqrt(emit_damping_rate_x),
    gauss_noise_ampl_y     = np.sqrt(physemit_y_eq*beta_y_sext_right) * np.sqrt(emit_damping_rate_y),
    gauss_noise_ampl_py    = np.sqrt(physemit_y_eq/beta_y_sext_right) * np.sqrt(emit_damping_rate_y),
    gauss_noise_ampl_zeta  =     sigma_z_eq * np.sqrt(emit_damping_rate_s),
    gauss_noise_ampl_pzeta = sigma_delta_eq * np.sqrt(emit_damping_rate_s),
)

# from left crab sextupole to IP2 (sy2l.1)
el_arc_right_b1 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beta_x_sext_right, beta_x],
    bety = [beta_y_sext_right, beta_y],
    alfx = [alpha_x_sext_right, 0],
    alfy = [alpha_y_sext_right, 0],
    bets = beta_s,
)

el_arc_left_b2 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beta_x_sext_left, beta_x],
    bety = [beta_y_sext_left, beta_y],
    alfx = [alpha_x_sext_left, 0],
    alfy = [alpha_y_sext_left, 0],
    bets = beta_s,
)

el_arc_mid_b2 = xt.LineSegmentMap(_context=context,
    qx =  qx,
    qy =  qy - 0.5, # subtract .25*2 phase advance from small arcs
    qs =  qs,
    betx = [beta_x_sext_right, beta_x_sext_left],
    bety = [beta_y_sext_right, beta_y_sext_left],
    alfx = [alpha_x_sext_right, alpha_x_sext_left],
    alfy = [alpha_y_sext_right, alpha_y_sext_left],
    bets = beta_s,
    damping_rate_x     = sigma_damping_rate_x,
    damping_rate_px    = sigma_damping_rate_x,
    damping_rate_y     = sigma_damping_rate_y,
    damping_rate_py    = sigma_damping_rate_y,
    damping_rate_zeta  = sigma_damping_rate_s,
    damping_rate_pzeta = sigma_damping_rate_s,

    # noise has to be normalized to beamsize at exit using beta at exit!
    gauss_noise_ampl_x     = np.sqrt(physemit_x_eq*beta_x_sext_left) * np.sqrt(emit_damping_rate_x),
    gauss_noise_ampl_px    = np.sqrt(physemit_x_eq/beta_x_sext_left) * np.sqrt(emit_damping_rate_x),
    gauss_noise_ampl_y     = np.sqrt(physemit_y_eq*beta_y_sext_left) * np.sqrt(emit_damping_rate_y),
    gauss_noise_ampl_py    = np.sqrt(physemit_y_eq/beta_y_sext_left) * np.sqrt(emit_damping_rate_y),
    gauss_noise_ampl_zeta  =     sigma_z_eq * np.sqrt(emit_damping_rate_s),
    gauss_noise_ampl_pzeta = sigma_delta_eq * np.sqrt(emit_damping_rate_s),
)

el_arc_right_b2 = xt.LineSegmentMap(_context=context,
    qx = 0,  # 2pi phase advance so integer part is zero
    qy = 0.25,  # 2.5pi
    qs = 0,  # no dipole here so no synchrotron motion
    betx = [beta_x, beta_x_sext_right],
    bety = [beta_y, beta_y_sext_right],
    alfx = [0, alpha_x_sext_right],
    alfy = [0, alpha_y_sext_right],
    bets = beta_s,
)

k2_left  = k2_factor / (2 * phi * beta_y * beta_y_sext_left ) * np.sqrt(beta_x / beta_x_sext_left )
k2_right = k2_factor / (2 * phi * beta_y * beta_y_sext_right) * np.sqrt(beta_x / beta_x_sext_right)

el_sextupole_left_b1  = xt.Multipole(order=2, knl=[0, 0,   k2_left])
el_sextupole_right_b1 = xt.Multipole(order=2, knl=[0, 0, -k2_right])
el_sextupole_left_b2  = xt.Multipole(order=2, knl=[0, 0,   k2_left])
el_sextupole_right_b2 = xt.Multipole(order=2, knl=[0, 0, -k2_right])

slicer = xf.TempSlicer(_context=context, n_slices=arg_n_slices, sigma_z=sigma_z_bs, mode="shatilov")

################################################################################
# the pipeline manager enables the communication of data between the two beams #
################################################################################

# identify the two beams as b1 and b2
particles_b1.init_pipeline('b1')
particles_b2.init_pipeline('b2')

# set up pipeline manager (needed for any simulations with communication between the two beams)
pipeline_manager = xt.PipelineManager()
pipeline_manager.add_particles('b1',0)
pipeline_manager.add_particles('b2',0)

# the communication of data (=the stat. moments) takes place at the element named IP1
pipeline_manager.add_element('IP1')

###########################################################
# config file needed as an input to the beambeam elements #
###########################################################

# this goes to bbeamIP1_b1
config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP1',
   partner_particles_name = 'b2',
   slicer=slicer,
   update_every=100,
   quasistrongstrong=True,
   )

# this goes to bbeamIP1_b2
config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP1',
   partner_particles_name = 'b1',
   slicer=slicer,
   update_every=100,
   quasistrongstrong=True,
   )

#################################
# define the beam-beam elements #
#################################

bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,

        ########################################################################
        # this is the charge in units of elementary charges of the strong beam #
        ########################################################################

        other_beam_q0 = particles_b2.q0,

        ##################################################################
        # phi=crossing angle in radians, alpha=crossing plane in radians #
        ##################################################################

        phi = phi,
        alpha=0,

        ##########################################################################################
        # the config object enabling the communication of moments to the other beam-beam element #
        ##########################################################################################

        config_for_update = config_for_update_b1_IP1,

        ########################################
        # record luminosity per bunch crossing #
        ########################################

        flag_luminosity=1,
)

bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
        _context=context,

        ########################################################################
        # this is the charge in units of elementary charges of the strong beam #
        ########################################################################

        other_beam_q0 = particles_b1.q0,

        ##################################################################
        # phi=crossing angle in radians, alpha=crossing plane in radians #
        ##################################################################

        phi = -phi, # -phi for beam 2!!!!
        alpha=0,

        ##########################################################################################
        # the config object enabling the communication of moments to the other beam-beam element #
        ##########################################################################################

        config_for_update = config_for_update_b2_IP1,

        ########################################
        # record luminosity per bunch crossing #
        ########################################

        flag_luminosity=1,
)

monitor_coords_b1_1 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=arg_n_turns, particle_id_range=(0,1000))
monitor_coords_b2_1 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=arg_n_turns, particle_id_range=(0,1000))
monitor_coords_b1_2 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=arg_n_turns, particle_id_range=(0,1000))
monitor_coords_b2_2 = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=arg_n_turns, particle_id_range=(0,1000))

line_b1 = xt.Line(elements = [
                              monitor_coords_b1_1,
                              bbeamIP1_b1,
                              el_arc_left_b1,
                              el_sextupole_left_b1,
                              el_arc_mid_b1,
                              monitor_coords_b1_2,
                              el_sextupole_right_b1,
                              el_arc_right_b1,

])

line_b2 = xt.Line(elements = [
                              monitor_coords_b2_1,
                              bbeamIP1_b2,
                              el_arc_right_b2,
                              el_sextupole_right_b2,
                              el_arc_mid_b2,
                              monitor_coords_b2_2,
                              el_sextupole_left_b2,
                              el_arc_left_b2,
])


line_b1.build_tracker(_context=context)
line_b2.build_tracker(_context=context)


assert line_b1._needs_rng == False
line_b1.configure_radiation(model=None, model_beamstrahlung='quantum')
line_b2.configure_radiation(model=None, model_beamstrahlung='quantum')
assert line_b1._needs_rng == True

branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
multitracker = xt.PipelineMultiTracker(branches=[branch_b1, branch_b2])

record_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D,
                                                      capacity={
                                                          "beamstrahlungtable": int(0),
                                                          "bhabhatable": int(0),
                                                          "lumitable": arg_n_turns,
                                                      })

record_b2 = line_b2.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D,
                                                      capacity={
                                                          "beamstrahlungtable": int(0),
                                                          "bhabhatable": int(0),
                                                          "lumitable": arg_n_turns,
                                                      })

#########
# track #
#########

# beam size files are on eos"
f_b1_name = "{}/coords_b1.txt".format(outputs_path)
f_b2_name = "{}/coords_b2.txt".format(outputs_path)
f_b1 = open(f_b1_name, "w")
f_b2 = open(f_b2_name, "w")

row_template = str(2*"{:.0f} "+13*"{:.6e} ")[:-1]+"\n"
f_b1.write("turn alive lumi x_avg px_avg y_avg py_avg z_avg delta_avg x_std px_std y_std py_std z_std delta_std\n")
f_b2.write("turn alive lumi x_avg px_avg y_avg py_avg z_avg delta_avg x_std px_std y_std py_std z_std delta_std\n")

prev_time = time.time()
for i in range(arg_n_turns):
    multitracker.track(num_turns=1)
    curr_time = time.time()
    if not i%10:
        print(f"Turn [{i+1}/{arg_n_turns}] took: {curr_time-prev_time} [s]")
    prev_time = curr_time

    alive_1     = np.sum(particles_b1.state[particles_b1.state==1])
    alive_2     = np.sum(particles_b2.state[particles_b2.state==1])

    # take lumi at particles.at_turn (start of turn particles.at_turn)
    i_b1 = 0
    i_b2 = 0
    while particles_b1.state[i_b1] != 1:
        i_b1 += 1
    while particles_b2.state[i_b2] != 1:
        i_b2 += 1
    at_turn_b1 = int(particles_b1.at_turn[i_b1]) - 1
    at_turn_b2 = int(particles_b2.at_turn[i_b2]) - 1
    lumi_b1 = record_b1.lumitable.luminosity[at_turn_b1]
    lumi_b2 = record_b2.lumitable.luminosity[at_turn_b2]

    x_avg_1     = np.mean(particles_b1.x    , axis=0)
    y_avg_1     = np.mean(particles_b1.y    , axis=0)
    px_avg_1    = np.mean(particles_b1.px   , axis=0)
    py_avg_1    = np.mean(particles_b1.py   , axis=0)
    z_avg_1     = np.mean(particles_b1.zeta , axis=0)
    delta_avg_1 = np.mean(particles_b1.delta, axis=0)

    x_avg_2     = np.mean(particles_b2.x    , axis=0)
    y_avg_2     = np.mean(particles_b2.y    , axis=0)
    px_avg_2    = np.mean(particles_b2.px   , axis=0)
    py_avg_2    = np.mean(particles_b2.py   , axis=0)
    z_avg_2     = np.mean(particles_b2.zeta , axis=0)
    delta_avg_2 = np.mean(particles_b2.delta, axis=0)

    x_std_1     = np.std(particles_b1.x    , axis=0)
    y_std_1     = np.std(particles_b1.y    , axis=0)
    px_std_1    = np.std(particles_b1.px   , axis=0)
    py_std_1    = np.std(particles_b1.py   , axis=0)
    z_std_1     = np.std(particles_b1.zeta , axis=0)
    delta_std_1 = np.std(particles_b1.delta, axis=0)

    x_std_2     = np.std(particles_b2.x    , axis=0)
    y_std_2     = np.std(particles_b2.y    , axis=0)
    px_std_2    = np.std(particles_b2.px   , axis=0)
    py_std_2    = np.std(particles_b2.py   , axis=0)
    z_std_2     = np.std(particles_b2.zeta , axis=0)
    delta_std_2 = np.std(particles_b2.delta, axis=0)

    # write row by row into file
    f_b1.write(row_template.format(i, int(alive_1), lumi_b1, x_avg_1, px_avg_1, y_avg_1, py_avg_1, z_avg_1, delta_avg_1, x_std_1, px_std_1, y_std_1, py_std_1, z_std_1, delta_std_1))
    f_b2.write(row_template.format(i, int(alive_2), lumi_b2, x_avg_2, px_avg_2, y_avg_2, py_avg_2, z_avg_2, delta_avg_2, x_std_2, px_std_2, y_std_2, py_std_2, z_std_2, delta_std_2))

line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

f_b1.close()
f_b2.close()

print("[exec.py] successfully finished simulation")
