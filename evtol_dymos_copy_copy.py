import openmdao.api as om
import dymos as dm
import numpy as np

import sys

sys.path.insert(0, "../ode")

from evtol_dynamics_comp import Dynamics

import verify_data

if __name__ == '__main__':
    input_arg_1 = 0.0

    input_arg_2 = 'ns'

    # Some specifications
    prop_rad = 0.75
    wing_S = 9.
    wing_span = 6.
    num_blades = 3.
    blade_chord = 0.1
    num_props = 8

    # User-specified input dictionary
    input_dict = {'T_guess': 9.8 * 725 * 1.2,  # initial thrust guess
                  'x_dot_initial': 0.,  # initial horizontal speed
                  'y_dot_initial': 0.01,  # initial vertical speed
                  'y_initial': 0.01,  # initial vertical displacement
                  'A_disk': np.pi * prop_rad ** 2 * num_props,  # total propeller disk area
                  'AR': wing_span ** 2 / (0.5 * wing_S),  # aspect ratio of each wing
                  'e': 0.68,  # span efficiency factor of each wing
                  't_over_c': 0.12,  # airfoil thickness-to-chord ratio
                  'S': wing_S,  # total wing reference area
                  'CD0': 0.35 / wing_S,  # coefficient of drag of the fuselage, gear, etc.
                  'm': 725.,  # mass of aircraft
                  'a0': 5.9,  # airfoil lift-curve slope
                  'alpha_stall': 15. / 180. * np.pi,  # wing stall angle
                  'rho': 1.225,  # air density
                  'induced_velocity_factor': int(input_arg_1) / 100.,  # induced-velocity factor
                  'stall_option': input_arg_2,  # stall option: 's' allows stall, 'ns' does not
                  'R': prop_rad,  # propeller radius
                  'solidity': num_blades * blade_chord / np.pi / prop_rad,  # solidity
                  'omega': 136. / prop_rad,  # angular rotation rate
                  'prop_CD0': 0.012,  # CD0 for prop profile power
                  'k_elec': 0.9,  # electrical and mechanical losses factor
                  'k_ind': 1.2,  # induced-losses factor
                  'nB': num_blades,  # number of blades per propeller
                  'bc': blade_chord,  # representative blade chord
                  'n_props': num_props  # number of propellers
                  }

    p = om.Problem()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    #####################################################
    # Phase-1 declaration                               #
    #####################################################
    ascent = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False),
                     ode_class=Dynamics,
                     ode_init_kwargs={'input_dict': input_dict})  ##the vectorized dynamics

    traj.add_phase('ascent', ascent)

    ascent.set_time_options(fix_initial=True, duration_bounds=(5, 60), duration_ref=30)
    ascent.add_state('x', fix_initial=True, rate_source='x_dot', ref0=0, ref=900, defect_ref=100)
    ascent.add_state('y', fix_initial=True, rate_source='y_dot', ref0=0, ref=300, defect_ref=300)
    ascent.add_state('vx', fix_initial=True, rate_source='a_x', ref0=0, ref=10)
    ascent.add_state('vy', fix_initial=True, rate_source='a_y', ref0=0, ref=10)
    ascent.add_state('energy', fix_initial=True, rate_source='energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    ascent.add_control('power', lower=1e3, upper=311000, ref0=1e3, ref=311000, rate_continuity=False)
    ascent.add_control('theta', lower=0., upper=3 * np.pi / 4, ref0=0, ref=3 * np.pi / 4,
                      rate_continuity=False)

    ascent.add_timeseries_output(['CL', 'CD'])
    # Boundary Constraints
    ascent.add_boundary_constraint('y', loc='final', lower=305,
                                  ref=100)  # Constraint for the final vertical displacement
    ascent.add_boundary_constraint('x', loc='final', equals=900,
                                  ref=100)  # Constraint for the final horizontal displacement
    ascent.add_boundary_constraint('x_dot', loc='final', equals=67.,
                                  ref=100)  # Constraint for the final horizontal speed

    # Path Constraints
    ascent.add_path_constraint('y', lower=0., upper=305,
                              ref=300)  # Constraint for the minimum vertical displacement
    ascent.add_path_constraint('acc', upper=0.3,
                              ref=1.0)  # Constraint for the acceleration magnitude
    ascent.add_path_constraint('aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
                              ref=np.radians(15))  # Constraint for the angle of attack
    ascent.add_path_constraint('thrust', lower=10, ref0=10,
                              ref=100)  # Constraint for the thrust magnitude

    #########################################################
    # Phase-2                                               #
    #########################################################
    descent = dm.Phase(transcription=dm.GaussLobatto(num_segments=10, order=3, solve_segments=False,
                                                   compressed=False),
                     ode_class=Dynamics,
                     ode_init_kwargs={'input_dict': input_dict})  ##the vectorized dynamics

    traj.add_phase('descent', descent)

    descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
    duration_ref=100, units='s')
    descent.add_state('x', fix_initial=False, fix_final=False, rate_source='x_dot') #, ref0=0, ref=900, defect_ref=100
    descent.add_state('y', fix_initial=False, fix_final=True, rate_source='y_dot') #, ref0=0, ref=300, defect_ref=300
    descent.add_state('vx', fix_initial=False, fix_final=True, rate_source='a_x', ref0=0, ref=10)
    descent.add_state('vy', fix_initial=True, fix_final=True, rate_source='a_y', ref0=0, ref=10)
    descent.add_state('energy', fix_initial=False, fix_final=False, rate_source='energy_dot', ref0=0, ref=1E7, defect_ref=1E5)

    descent.add_control('power', lower=1e3, upper=311000, ref0=1e3, ref=311000, rate_continuity=False)
    descent.add_control('theta', lower=0., upper=3 * np.pi / 4, ref0=0, ref=3 * np.pi / 4,
                      rate_continuity=False)

    descent.add_timeseries_output(['CL', 'CD'])
    # Link Phases (link time and all state variables)
    traj.link_phases(phases=['ascent', 'descent'], vars=['*'])
    

    # Boundary Constraints
    descent.add_boundary_constraint('y', loc='final', lower=1,
                                  ref=100)  # Constraint for the final vertical displacement
    descent.add_boundary_constraint('x', loc='final', equals=1800,
                                  ref=100)  # Constraint for the final horizontal displacement
    descent.add_boundary_constraint('x_dot', loc='final', equals=1.,
                                  ref=100)  # Constraint for the final horizontal speed

    # Path Constraints
    descent.add_path_constraint('y', lower=0.1, upper=305,
                              ref=300)  # Constraint for the minimum vertical displacement
    descent.add_path_constraint('acc', upper=0.01,
                              ref=1.0)  # Constraint for the acceleration magnitude
    descent.add_path_constraint('aoa', lower=-np.radians(15), upper=np.radians(15), ref0=-np.radians(15),
                              ref=np.radians(15))  # Constraint for the angle of attack
    descent.add_path_constraint('thrust', lower=-10, ref0=10,
                              ref=100)  # Constraint for the thrust magnitude
    
    # Objective
    descent.add_objective('energy', loc='final', ref0=0, ref=1E7)
    # # Setup the driver
    p.driver = om.pyOptSparseDriver()

    # p.driver.options['optimizer'] = 'SNOPT'
    # p.driver.opt_settings['Major optimality tolerance'] = 1e-4
    # p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    # p.driver.opt_settings['Major iterations limit'] = 1000
    # p.driver.opt_settings['Minor iterations limit'] = 100_000_000
    # p.driver.opt_settings['iSumm'] = 6

    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.opt_settings['max_iter'] = 1000
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['tol'] = 5.0E-5

    p.driver.declare_coloring(tol=1.0E-8)

    p.setup()
    # Set Initial Values for ascent Phase
    p.set_val('traj.ascent.t_initial', 0.0)
    p.set_val('traj.ascent.t_duration', 30)
    p.set_val('traj.ascent.states:x', ascent.interpolate(ys=[0, 900], nodes='state_input'))
    p.set_val('traj.ascent.states:y', ascent.interpolate(ys=[0.01, 300], nodes='state_input'))
    p.set_val('traj.ascent.states:vx', ascent.interpolate(ys=[0, 60], nodes='state_input'))
    p.set_val('traj.ascent.states:vy', ascent.interpolate(ys=[0.01, 10], nodes='state_input'))
    p.set_val('traj.ascent.states:energy', ascent.interpolate(ys=[0, 1E7], nodes='state_input'))

    p.set_val('traj.ascent.controls:power', ascent.interpolate(xs=np.linspace(0, 28.368, 500),
                                                              ys=verify_data.powers.ravel(),
                                                              nodes='control_input'))
    p.set_val('traj.ascent.controls:theta', ascent.interpolate(xs=np.linspace(0, 28.368, 500),
                                                              ys=verify_data.thetas.ravel(),
                                                              nodes='control_input'))

    p.set_val('traj.ascent.controls:power', 200000.0)
    p.set_val('traj.ascent.controls:theta', ascent.interpolate(ys=[0.001, np.radians(85)], nodes='control_input'))
    # Set Initial Values for descent Phase
    p.set_val('traj.descent.t_initial', 30.0)
    p.set_val('traj.descent.t_duration', 30)
    p.set_val('traj.descent.states:x', descent.interpolate(ys=[900, 1800], nodes='state_input'))
    p.set_val('traj.descent.states:y', descent.interpolate(ys=[300, 600], nodes='state_input'))
    p.set_val('traj.descent.states:vx', descent.interpolate(ys=[60, 120], nodes='state_input'))
    p.set_val('traj.descent.states:vy', descent.interpolate(ys=[10, 20], nodes='state_input'))
    p.set_val('traj.descent.states:energy', descent.interpolate(ys=[1E7, 1E7 ], nodes='state_input'))

    p.set_val('traj.descent.controls:power', descent.interpolate(xs=np.linspace(28.368, 2*28.368, 500),
                                                              ys=verify_data.powers.ravel(),
                                                              nodes='control_input'))
    p.set_val('traj.descent.controls:theta', descent.interpolate(xs=np.linspace(28.368, 2*28.368, 500),
                                                              ys=verify_data.thetas.ravel(),
                                                              nodes='control_input'))

    p.set_val('traj.descent.controls:power', 200000.0)
    p.set_val('traj.descent.controls:theta', descent.interpolate(ys=[0.001,np.radians(85) ], nodes='control_input'))

    dm.run_problem(p, run_driver=True, simulate=True)
