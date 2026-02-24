#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2026/02/24 01:37:36
#


import pytest
from src import spindle_state as ss
from src import lattice as lat
import numpy as np
import os
import pickle

test_spindle_lattice = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
])

expected_mt_vecs = np.array([[ 0.5,  0. ,  0. ],
       [-1.5,  0. ,  0. ],
       [-0.5,  1. ,  0. ],
       [-0.5, -1. ,  0. ],
       [-0.5,  0. ,  1. ],
       [-0.5,  0. , -1. ]])

expected_mt_dirs = np.array([[1.000, 0.000, 0.000],
       [-1.000, 0.000, 0.000],
       [-0.447, 0.894, 0.000],
       [-0.447, -0.894, 0.000],
       [-0.447, 0.000, 0.894],
       [-0.447, 0.000, -0.894]])

expected_mt_norms = np.array([0.5, 1.5, 1.11803399, 1.11803399, 1.11803399, 1.11803399])

test_spindle_state = np.array([1, 1, 3, 3, 1, 1])

initial_mtoc_pos = np.array([0.5, 0, 0])

# test_spindle = ss.Spindle(initial_mtoc_pos, test_spindle_state, test_spindle_lattice)

test_spindle_dir = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/tests/test_spindle/'


def test_normalize_vecs():
    # one vector
    vec = np.array([0,0,0.5])

    normalized_vec, norm = ss.normalize_vecs(vec)

    assert norm == 0.5

    assert (normalized_vec == np.array([0,0,1])).all()

    #multiple vectors
    vecs=np.array([[0,0,0.5], [0.5, 0, 0]])

    normalized_vecs, norms = ss.normalize_vecs(vecs)

    assert (norms == np.array([0.5, 0.5])).all()

    assert (normalized_vecs == np.array([[0., 0., 1.], [1., 0., 0.]])).all()


def test_circle_lattice():
    assert (np.round(ss.normalize_vecs(lat.circle_lattice(10))[1],5) == 1.0).all()


def test_spindle_state_init():
    test_spindle = ss.Spindle(
            initial_mtoc_pos,  #
            initial_spindle_state = test_spindle_state,  #
            lattice_sites=test_spindle_lattice,
            seed=1,
            dir_path = test_spindle_dir,
            save=True,
            readout=False,
    )
    ## testing basic attributes and init for Class spindle

    assert (test_spindle.mtoc_pos == np.array([0.5, 0, 0])).all()

    assert (test_spindle.spindle_state == test_spindle_state).all()

    assert (test_spindle.lattice_sites == test_spindle_lattice).all()

    assert (test_spindle.boundary_unit_normals == test_spindle_lattice).all() # true for any lattice on a unit sphere

    assert (test_spindle.mt_vecs == expected_mt_vecs).all()

    assert (test_spindle.mt_dirs - expected_mt_dirs < 0.001).all()

    assert test_spindle.num_sites == 6

    assert (np.round(test_spindle.mt_norms, 5) == np.round(expected_mt_norms, 5)).all()

    assert test_spindle.timestep_size == 0.001

    assert test_spindle.acceptable_radius == 0.05

    assert test_spindle.f_pull_0 == 1

    assert test_spindle.rigidity == 1

    assert test_spindle.friction_coefficient == 1

    assert test_spindle.growth_rate == 1

    assert test_spindle.stall_force == 1

    assert test_spindle.drag_factor == 100

    assert test_spindle.boundary_radius == 1

    assert test_spindle.tubulin_budget == 4

    assert test_spindle.dir_path == test_spindle_dir

    assert test_spindle.seed == 1 

    assert test_spindle.current_time == 0

    assert test_spindle.trajectory == {}

    assert test_spindle.spindle_trace.shape == (1,6)


def test_as_dict():

    test_spindle = ss.Spindle(
            initial_mtoc_pos,  #
            initial_spindle_state = test_spindle_state,  #
            lattice_sites=test_spindle_lattice,
            seed=1,
            dir_path = test_spindle_dir,
            save=True,
            readout=False,
    )
    spindle_dict = test_spindle.as_dict()    
    # -- parameters
    assert (spindle_dict['mtoc_pos'] == test_spindle.mtoc_pos).all()
    assert (spindle_dict['spindle_state'] == test_spindle.empty_spindle_state).all()
    assert (spindle_dict['lattice_sites'] == test_spindle.lattice_sites).all()

    # -- hyperparameters --
    assert spindle_dict['timestep_size'] == test_spindle.timestep_size
    assert spindle_dict['acceptable_radius'] == test_spindle.acceptable_radius
    assert spindle_dict['f_pull_0'] == test_spindle.f_pull_0
    assert spindle_dict['rigidity'] == test_spindle.rigidity
    assert spindle_dict['friction_coefficient'] == test_spindle.friction_coefficient
    assert spindle_dict['growth_rate'] == test_spindle.growth_rate
    assert spindle_dict['stall_force'] == test_spindle.stall_force
    assert spindle_dict['drag_factor'] == test_spindle.drag_factor
    assert spindle_dict['boundary_radius'] == test_spindle.boundary_radius
    assert spindle_dict['tubulin_budget'] == test_spindle.tubulin_budget
    assert spindle_dict['dir_path'] == test_spindle.dir_path
    assert spindle_dict['seed'] == test_spindle.seed

def test_init_spindle_from_dict():
    spindle_path = os.path.join(test_spindle_dir, 'spindle.pkl')
    with open(spindle_path, 'rb') as f:
        spindle_dict = pickle.load(f)
    
    spindle_from_dict = ss.init_spindle_from_dict(spindle_dict)

    assert (spindle_from_dict.mtoc_pos == np.array([0.5, 0, 0])).all()

    assert (spindle_from_dict.spindle_state == test_spindle_state).all()

    assert (spindle_from_dict.lattice_sites == test_spindle_lattice).all()

    assert (spindle_from_dict.boundary_unit_normals == test_spindle_lattice).all() # true for any lattice on a unit sphere

    assert (spindle_from_dict.mt_vecs == expected_mt_vecs).all()

    assert (spindle_from_dict.mt_dirs - expected_mt_dirs < 0.001).all()

    assert spindle_from_dict.num_sites == 6

    assert (np.round(spindle_from_dict.mt_norms, 5) == np.round(expected_mt_norms, 5)).all()

    assert spindle_from_dict.timestep_size == 0.001

    assert spindle_from_dict.acceptable_radius == 0.05

    assert spindle_from_dict.f_pull_0 == 1

    assert spindle_from_dict.rigidity == 1

    assert spindle_from_dict.friction_coefficient == 1

    assert spindle_from_dict.growth_rate == 1

    assert spindle_from_dict.stall_force == 1

    assert spindle_from_dict.drag_factor == 100

    assert spindle_from_dict.boundary_radius == 1

    assert spindle_from_dict.tubulin_budget == 4

    assert spindle_from_dict.dir_path == test_spindle_dir

    assert spindle_from_dict.seed == 1 

    assert spindle_from_dict.current_time == 0

    assert spindle_from_dict.trajectory == {}

    assert spindle_from_dict.spindle_trace.shape == (1,6)

def test_set_mtoc_pos():
    test_spindle = ss.Spindle(
            initial_mtoc_pos,  #
            initial_spindle_state = test_spindle_state,  #
            lattice_sites=test_spindle_lattice,
            seed=1,
            dir_path = test_spindle_dir,
            save=True,
            readout=False,
    )
    
    new_mtoc_pos = np.array([0, 0.5, 0])

    test_spindle.set_mtoc_pos(new_mtoc_pos)

    assert (test_spindle.mtoc_pos == new_mtoc_pos).all()


def test_add_remove_microtubules():
    test_spindle = ss.Spindle(
            initial_mtoc_pos,  #
            initial_spindle_state = test_spindle_state,  #
            lattice_sites=test_spindle_lattice,
            seed=1,
            dir_path = test_spindle_dir,
            save=True,
            readout=False,
    )
    ## testing add and remove MTs

    # add MTs
    test_mt_indices_to_add = np.array([1,2])

    test_spindle.add_microtubules(test_mt_indices_to_add)

    assert (test_spindle.spindle_state == np.array([1., 2., 4., 3., 1., 1.])).all()

    # try to add the same MTs again
    with pytest.raises(ValueError):
        test_spindle.add_microtubules(test_mt_indices_to_add)
    
    # remove the added MTs
    test_spindle.remove_microtubules(test_mt_indices_to_add)

    assert (test_spindle.spindle_state == np.array([1., 1., 3., 3., 1., 1.])).all()

    # try to remove the same MTs again
    with pytest.raises(ValueError):
        test_spindle.remove_microtubules(test_mt_indices_to_add)

    old_spindle = np.copy(test_spindle.spindle_state)

    test_spindle.add_microtubules(np.array([]))

    assert (test_spindle.spindle_state == old_spindle).all()

    test_spindle.remove_microtubules(np.array([]))

    assert (test_spindle.spindle_state == old_spindle).all()


def test_update_spindle():
    spindle_path = os.path.join(test_spindle_dir, 'spindle.pkl')
    with open(spindle_path, 'rb') as f:
        spindle_dict = pickle.load(f)
    
    spindle_from_dict = ss.init_spindle_from_dict(spindle_dict)

    new_spindle_state = np.array([1, 2, 3, 4, 1, 2])

    spindle_from_dict.update_spindle_state(new_spindle_state=new_spindle_state)

    assert (spindle_from_dict.spindle_state == new_spindle_state).all()


def test_relax_pulling():
    # making a spindle with a circular
    circ_2_lattice = lat.circle_lattice(2)
    circ_2_spindle_state = np.array([4,4])
    initial_circ_2_pos = np.array([0.2, 0.4, 0])

    circ_2 = ss.Spindle(initial_circ_2_pos, circ_2_spindle_state, circ_2_lattice,)
    circ_2.relax()

    (np.round(circ_2.mtoc_pos, 3) == np.array([0.185, 0.000, 0.000])).all()

    circ_4_lattice = lat.circle_lattice(4)
    circ_4_spindle_state = np.array([4,4,4,4])
    initial_circ_4_pos = np.array([0.2, 0.4, 0])

    circ_4 = ss.Spindle(initial_circ_4_pos, circ_4_spindle_state, circ_4_lattice,)

    circ_4.relax(resolution=5)

    assert (np.round(circ_4.mtoc_pos, 3) == np.array([0.0, 0.0, 0.0])).all()


def test_restart_experiment():
    dir_path = '/Users/emrealca/Documents/Penn/flatiron-microtubules/simulations/tests/testing_reconstruction_2026-02-24_01-09-05'

    spindle_from_dir = ss.restart_experiment_from_directory(dir_path, readout=True)

    assert (np.round(spindle_from_dir.mtoc_pos, 3) == np.array([-0.026, 0.001, 0.000])).all()
    assert np.round(spindle_from_dir.current_time, 3) == 351.588
    assert np.round(spindle_from_dir.calc_cost(), 3) == 0.040
        

# def test_calculate_pulling_forces():

#     # add two pulling MTs opposing each other
#     test_spindle.add_microtubules([2,3])
#     assert (test_spindle.spindle_state == np.array([1., 1., 4., 4., 1., 1.])).all()

#     # when in the orthogonal plane to the two pulling MTs, restoring force points directly back to origin
#     new_mtoc_pos = np.array([0.5, 0, 0])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (np.round(test_spindle.calculate_pulling_forces(), 5) == np.round(np.array([-0.89442719, 0.,  0.]), 5)).all()

#     new_mtoc_pos = np.array([0, 0, 0.5])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (np.round(test_spindle.calculate_pulling_forces()) == np.round(np.array([0., 0.,  -0.89442719]))).all()

#     # when in the same axis as the MTs, no net force
#     new_mtoc_pos = np.array([0, 0.5, 0])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (test_spindle.calculate_pulling_forces() == np.array([0., 0.,  0])).all()
#     # assert ((test_spindle.calculate_pulling_forces() - np.array([0., 0.,  0])) < tolerance * np.ones(3)).all()

#     # when equally away from the origin on the 0-plane in both coordinates orthogonal to the pulling MTs, pull equally on both
#     new_mtoc_pos = np.array([0.5, 0, 0.5])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (np.round(test_spindle.calculate_pulling_forces()) == np.round(np.array([-0.81649658,0., -0.81649658]))).all()

#     # remove MTs for whatever the next test is
#     test_spindle.remove_microtubules([2,3])
#     assert (test_spindle.spindle_state == np.array([1., 1., 3., 3., 1., 1.])).all()


# def test_calculate_pushing_forces():

#     # set 2 MT's pushing along the same axis opposing each other
#     test_spindle.add_microtubules(np.array([0,1]))

#     # no net force when we are directly between two MTs pushing perpendicular to the boundary
#     test_spindle.set_mtoc_pos(np.array([0,0,0]))
#     assert (test_spindle.calculate_pushing_forces() == np.array([0., 0., 0.,])).all()
    
#     # no net force anywhere along the same axis as 2 opposing pushing MTs without buckling
#     test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
#     assert (test_spindle.calculate_pushing_forces() == np.array([0., 0., 0.,])).all()

#     # if the mtoc so close to one side that one mt buckles, the force of the near mt should provide a restoring force
#     test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
#     assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0.03037264, 0., 0.,]), 5)).all()

#     # when in a position in the orthogonal plane to the two pushing mts, the pushing force is expected to be destabilizing in that orthogonal plane
#     test_spindle.set_mtoc_pos(np.array([0,0.5,0]))
#     assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0., 0.80901699, -0.]), 5)).all()

#     test_spindle.set_mtoc_pos(np.array([0,0.5,0.5]))
#     assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0., 0.68989795, 0.68989795]), 5)).all()

#     # remove MTs for whatever the next test is
#     test_spindle.remove_microtubules([0,1])
#     assert (test_spindle.spindle_state == np.array([1., 1., 3., 3., 1., 1.])).all()

# def test_calc_mtoc_velocity():

#     # -- pulling mts only --
#     # set 2 MT's pulling along the same axis opposing each other
#     test_spindle.add_microtubules([2,3])

#     # zero net force --> zero velocity
#     new_mtoc_pos = np.array([0, 0.5, 0])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (test_spindle.calc_mtoc_velocity() == np.array([0,0,0])).all()

#     # stabilizing force --> stabilizing velocity
#     new_mtoc_pos = np.array([0.5, 0, 0.5])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.00816497, 0., -0.00816497]), 5)).all()

#     # destabilizing force --> destabilizing velocity
#     # with only 1 pulling MT, the pulling force will always be destabilizing
#     test_spindle.remove_microtubules([3])
#     new_mtoc_pos = np.array([0, 0, 0])
#     test_spindle.set_mtoc_pos(new_mtoc_pos)
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([0., 0.01, 0.]), 5)).all()

#     # remove MTs for next test
#     test_spindle.remove_microtubules([2])

#     # -- pushing only --
#     # set 2 MT's pushing along the same axis opposing each other
#     test_spindle.add_microtubules(np.array([0,1]))

#     # zero net force --> zero velocity
#     test_spindle.set_mtoc_pos(np.array([0,0,0]))
#     assert (test_spindle.calc_mtoc_velocity() == np.array([0., 0., 0.,])).all()

#     # stabilizing force --> stabilizing velocity
#     test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.00030373, 0., 0.,]), 5)).all()

#     # destabilizing force --> destabilizing velocity
#     test_spindle.set_mtoc_pos(np.array([0,0.5,0.5]))
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0., 0.00689897, 0.00689897]), 5)).all()

#     # -- both pushing and pulling MTs --

#     # adding pulling MTs
#     test_spindle.add_microtubules([2,3])

#     # zero net force --> zero velocity
#     test_spindle.set_mtoc_pos(np.array([0,0,0]))
#     assert (test_spindle.calc_mtoc_velocity() == np.array([0., 0., 0.,])).all()

#     # stabilizing force --> stabilizing velocity
#     test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.01279763, 0., 0.]), 5)).all()

#     # destabilizing force --> destabilizing velocity
#     test_spindle.remove_microtubules([3])
#     test_spindle.set_mtoc_pos(np.array([0,0,0.5]))
#     assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([0., 0.00894427, 0.00361803]), 5)).all()

#     # remove MTs for next test
#     test_spindle.remove_microtubules([0,1, 2])
#     assert (test_spindle.spindle_state == np.array([1, 1, 3, 3, 1, 1])).all()

# def test_time_evolution():
#     # -- basic motion --
#     # adding pulling MTs
#     test_spindle.add_microtubules([0, 1, 2, 3])

#     # zero net force --> zero change in position
#     old_mtoc_pos = np.array([0,0,0])
#     test_spindle.set_mtoc_pos(old_mtoc_pos)
#     new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
#     # assert (np.round(new_mtoc_pos,5) == np.round(old_mtoc_pos, 5)).all()
#     assert np.round(ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1], 5) == 0
#     assert boundary_violated == False

#     # # stabilizing force --> shorter mtoc_pos norm
#     old_mtoc_pos = np.array([0.8,0,0])
#     test_spindle.set_mtoc_pos(old_mtoc_pos)
#     new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
#     assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] < 0
#     assert boundary_violated == False

#     # # destabilizing force --> longer mtoc_pos norm
#     test_spindle.remove_microtubules([3])
#     old_mtoc_pos = np.array([0,0,0.5])
#     test_spindle.set_mtoc_pos(old_mtoc_pos)
#     new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
#     assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] > 0
#     assert boundary_violated == False

#     # -- boundary violation
#     # violate boundary and ensure that the proper vector correction occurs and that boundary_violated is set to true
#     old_mtoc_pos = np.array([0,0,0.999999])
#     test_spindle.set_mtoc_pos(old_mtoc_pos)
#     new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
#     assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] > 0
#     assert boundary_violated == True

#     # remove MTs for next test
#     test_spindle.remove_microtubules([0, 1, 2])
#     assert (test_spindle.spindle_state == np.array([1, 1, 3, 3, 1, 1])).all()


# def test_calc_cost():
#     # further from the origin -> higher cost, even in the absense of MTs

#     test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
#     assert test_spindle.calc_cost() == 0.25

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
#     assert np.round(test_spindle.calc_cost(), 5) == 0.5

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
#     assert np.round(test_spindle.calc_cost(),5) == 0.75

#     # adding MTs does not affect cost so long as their total sum distance is less than the allowed maximum (6 units here)

#     test_spindle.add_microtubules([0,1,2,3])

#     test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
#     assert test_spindle.calc_cost() == 0.25

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
#     assert np.round(test_spindle.calc_cost(), 5) == 0.5

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
#     assert np.round(test_spindle.calc_cost(),5) == 0.75

#     # adding MTs beyond the threshold does contribute to cost

#     test_spindle.add_microtubules([4,5])

#     test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
#     np.round(test_spindle.calc_cost(), 5) == 0.29969

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
#     assert np.round(test_spindle.calc_cost(), 5) == 1.60804

#     test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
#     assert np.round(test_spindle.calc_cost(),5) == 6.87251

#     test_spindle.remove_microtubules([0, 1, 2, 3, 4, 5])