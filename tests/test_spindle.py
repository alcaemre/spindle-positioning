#
# Emre Alca
# University of Pennsylvania
# Created on Sat Nov 22 2025
# Last Modified: 2026/01/10 17:58:35
#


import pytest
from src import spindle_state as ss
import numpy as np

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

expected_mt_norms = np.array([0.5, 1.5, 1.11803399, 1.11803399, 1.11803399, 1.11803399])

test_spindle_state = np.array([1, 1, 3, 3, 1, 1])

initial_mtoc_pos = np.array([0.5, 0, 0])

test_spindle = ss.Spindle(initial_mtoc_pos, test_spindle_state, test_spindle_lattice)


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


def test_spindle_state_init():
    ## testing basic attributes and init for Class spindle

    assert (test_spindle.mtoc_pos == np.array([0.5, 0, 0])).all()

    assert (test_spindle.spindle_state == test_spindle_state).all()

    assert (test_spindle.lattice_sites == test_spindle_lattice).all()

    assert (test_spindle.boundary_unit_normals == test_spindle_lattice).all() # true for any lattice on a unit sphere

    assert (test_spindle.mt_vecs == expected_mt_vecs).all()

    assert test_spindle.num_sites == 6

    assert (np.round(test_spindle.mt_norms, 5) == np.round(expected_mt_norms, 5)).all()

    assert (np.round(test_spindle.max_total_mt_length, 5) == np.round(6, 5)).all()

    assert test_spindle.mt_len_cost_punishment_degree == 4

    assert test_spindle.cytoplasmic_catastrophe_rate == 1


def test_set_mtoc_pos():
    new_mtoc_pos = np.array([0, 0.5, 0])

    test_spindle.set_mtoc_pos(new_mtoc_pos)

    assert (test_spindle.mtoc_pos == new_mtoc_pos).all()


def test_add_remove_microtubules():
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
        

def test_calculate_pulling_forces():

    # add two pulling MTs opposing each other
    test_spindle.add_microtubules([2,3])
    assert (test_spindle.spindle_state == np.array([1., 1., 4., 4., 1., 1.])).all()

    # when in the orthogonal plane to the two pulling MTs, restoring force points directly back to origin
    new_mtoc_pos = np.array([0.5, 0, 0])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (np.round(test_spindle.calculate_pulling_forces(), 5) == np.round(np.array([-0.89442719, 0.,  0.]), 5)).all()

    new_mtoc_pos = np.array([0, 0, 0.5])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (np.round(test_spindle.calculate_pulling_forces()) == np.round(np.array([0., 0.,  -0.89442719]))).all()

    # when in the same axis as the MTs, no net force
    new_mtoc_pos = np.array([0, 0.5, 0])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (test_spindle.calculate_pulling_forces() == np.array([0., 0.,  0])).all()
    # assert ((test_spindle.calculate_pulling_forces() - np.array([0., 0.,  0])) < tolerance * np.ones(3)).all()

    # when equally away from the origin on the 0-plane in both coordinates orthogonal to the pulling MTs, pull equally on both
    new_mtoc_pos = np.array([0.5, 0, 0.5])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (np.round(test_spindle.calculate_pulling_forces()) == np.round(np.array([-0.81649658,0., -0.81649658]))).all()

    # remove MTs for whatever the next test is
    test_spindle.remove_microtubules([2,3])
    assert (test_spindle.spindle_state == np.array([1., 1., 3., 3., 1., 1.])).all()


def test_calculate_pushing_forces():

    # set 2 MT's pushing along the same axis opposing each other
    test_spindle.add_microtubules(np.array([0,1]))

    # no net force when we are directly between two MTs pushing perpendicular to the boundary
    test_spindle.set_mtoc_pos(np.array([0,0,0]))
    assert (test_spindle.calculate_pushing_forces() == np.array([0., 0., 0.,])).all()
    
    # no net force anywhere along the same axis as 2 opposing pushing MTs without buckling
    test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
    assert (test_spindle.calculate_pushing_forces() == np.array([0., 0., 0.,])).all()

    # if the mtoc so close to one side that one mt buckles, the force of the near mt should provide a restoring force
    test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
    assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0.03037264, 0., 0.,]), 5)).all()

    # when in a position in the orthogonal plane to the two pushing mts, the pushing force is expected to be destabilizing in that orthogonal plane
    test_spindle.set_mtoc_pos(np.array([0,0.5,0]))
    assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0., 0.80901699, -0.]), 5)).all()

    test_spindle.set_mtoc_pos(np.array([0,0.5,0.5]))
    assert (np.round(test_spindle.calculate_pushing_forces(),5) == np.round(np.array([-0., 0.68989795, 0.68989795]), 5)).all()

    # remove MTs for whatever the next test is
    test_spindle.remove_microtubules([0,1])
    assert (test_spindle.spindle_state == np.array([1., 1., 3., 3., 1., 1.])).all()

def test_calc_mtoc_velocity():

    # -- pulling mts only --
    # set 2 MT's pulling along the same axis opposing each other
    test_spindle.add_microtubules([2,3])

    # zero net force --> zero velocity
    new_mtoc_pos = np.array([0, 0.5, 0])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (test_spindle.calc_mtoc_velocity() == np.array([0,0,0])).all()

    # stabilizing force --> stabilizing velocity
    new_mtoc_pos = np.array([0.5, 0, 0.5])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.00816497, 0., -0.00816497]), 5)).all()

    # destabilizing force --> destabilizing velocity
    # with only 1 pulling MT, the pulling force will always be destabilizing
    test_spindle.remove_microtubules([3])
    new_mtoc_pos = np.array([0, 0, 0])
    test_spindle.set_mtoc_pos(new_mtoc_pos)
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([0., 0.01, 0.]), 5)).all()

    # remove MTs for next test
    test_spindle.remove_microtubules([2])

    # -- pushing only --
    # set 2 MT's pushing along the same axis opposing each other
    test_spindle.add_microtubules(np.array([0,1]))

    # zero net force --> zero velocity
    test_spindle.set_mtoc_pos(np.array([0,0,0]))
    assert (test_spindle.calc_mtoc_velocity() == np.array([0., 0., 0.,])).all()

    # stabilizing force --> stabilizing velocity
    test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.00030373, 0., 0.,]), 5)).all()

    # destabilizing force --> destabilizing velocity
    test_spindle.set_mtoc_pos(np.array([0,0.5,0.5]))
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0., 0.00689897, 0.00689897]), 5)).all()

    # -- both pushing and pulling MTs --

    # adding pulling MTs
    test_spindle.add_microtubules([2,3])

    # zero net force --> zero velocity
    test_spindle.set_mtoc_pos(np.array([0,0,0]))
    assert (test_spindle.calc_mtoc_velocity() == np.array([0., 0., 0.,])).all()

    # stabilizing force --> stabilizing velocity
    test_spindle.set_mtoc_pos(np.array([0.8,0,0]))
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([-0.01279763, 0., 0.]), 5)).all()

    # destabilizing force --> destabilizing velocity
    test_spindle.remove_microtubules([3])
    test_spindle.set_mtoc_pos(np.array([0,0,0.5]))
    assert (np.round(test_spindle.calc_mtoc_velocity(),5) == np.round(np.array([0., 0.00894427, 0.00361803]), 5)).all()

    # remove MTs for next test
    test_spindle.remove_microtubules([0,1, 2])
    assert (test_spindle.spindle_state == np.array([1, 1, 3, 3, 1, 1])).all()

def test_time_evolution():
    # -- basic motion --
    # adding pulling MTs
    test_spindle.add_microtubules([0, 1, 2, 3])

    # zero net force --> zero change in position
    old_mtoc_pos = np.array([0,0,0])
    test_spindle.set_mtoc_pos(old_mtoc_pos)
    new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
    # assert (np.round(new_mtoc_pos,5) == np.round(old_mtoc_pos, 5)).all()
    assert np.round(ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1], 5) == 0
    assert boundary_violated == False

    # # stabilizing force --> shorter mtoc_pos norm
    old_mtoc_pos = np.array([0.8,0,0])
    test_spindle.set_mtoc_pos(old_mtoc_pos)
    new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
    assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] < 0
    assert boundary_violated == False

    # # destabilizing force --> longer mtoc_pos norm
    test_spindle.remove_microtubules([3])
    old_mtoc_pos = np.array([0,0,0.5])
    test_spindle.set_mtoc_pos(old_mtoc_pos)
    new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
    assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] > 0
    assert boundary_violated == False

    # -- boundary violation
    # violate boundary and ensure that the proper vector correction occurs and that boundary_violated is set to true
    old_mtoc_pos = np.array([0,0,0.999999])
    test_spindle.set_mtoc_pos(old_mtoc_pos)
    new_mtoc_pos, boundary_violated = test_spindle.mtoc_time_evolution()
    assert ss.normalize_vecs(new_mtoc_pos)[1] - ss.normalize_vecs(old_mtoc_pos)[1] > 0
    assert boundary_violated == True

    # remove MTs for next test
    test_spindle.remove_microtubules([0, 1, 2])
    assert (test_spindle.spindle_state == np.array([1, 1, 3, 3, 1, 1])).all()


def test_calc_cost():
    # further from the origin -> higher cost, even in the absense of MTs

    test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
    assert test_spindle.calc_cost() == 0.25

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
    assert np.round(test_spindle.calc_cost(), 5) == 0.5

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
    assert np.round(test_spindle.calc_cost(),5) == 0.75

    # adding MTs does not affect cost so long as their total sum distance is less than the allowed maximum (6 units here)

    test_spindle.add_microtubules([0,1,2,3])

    test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
    assert test_spindle.calc_cost() == 0.25

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
    assert np.round(test_spindle.calc_cost(), 5) == 0.5

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
    assert np.round(test_spindle.calc_cost(),5) == 0.75

    # adding MTs beyond the threshold does contribute to cost

    test_spindle.add_microtubules([4,5])

    test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
    np.round(test_spindle.calc_cost(), 5) == 0.29969

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0]))
    assert np.round(test_spindle.calc_cost(), 5) == 1.60804

    test_spindle.set_mtoc_pos(np.array([0.5,0.5,0.5]))
    assert np.round(test_spindle.calc_cost(),5) == 6.87251

    test_spindle.remove_microtubules([0, 1, 2, 3, 4, 5])


def test_biased_spatial_nucleation_distribution():

    # tests with no MTs already present

    test_spindle.set_mtoc_pos(np.array([0,0,0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.31831, 0.31831, 0.31831, 0.31831, 0.31831, 0.31831]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0.5,0,0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.63662, 0.     , 0.46066, 0.46066, 0.17596, 0.17596]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0,0.5,0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.17596, 0.17596, 0.     , 0.63662, 0.17596, 0.17596]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0,0,0.5]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.17596, 0.17596, 0.46066, 0.46066, 0.63662, 0.     ]) == 0).all()

    # tests with some MTs present

    test_spindle.add_microtubules([0, 3])

    test_spindle.set_mtoc_pos(np.array([0, 0.5, 0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.     , 0.17596, 0.     , 0.     , 0.17596, 0.17596]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0, 0, 0.5]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.     , 0.17596, 0.46066, 0.     , 0.63662, 0.     ]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0.     , 0.31831, 0.31831, 0.     , 0.31831, 0.31831]) == 0).all()

    # test with all MTS present

    test_spindle.add_microtubules([ 1, 2, 4, 5])

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_spatial_nucleation_distribution().round(5) - np.array([0., 0., 0., 0., 0., 0.]) == 0).all()

    test_spindle.remove_microtubules([0,1,2,3,4,5])

def test_biased_spatial_catastrophe_distribution():

    # test with no MTs present

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0., 0., 0., 0., 0., 0.]) == 0).all()

    # tests with all MTs present

    test_spindle.add_microtubules([0, 1, 2, 3, 4, 5])

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.31831, 0.31831, 0.31831, 0.31831, 0.31831, 0.31831]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0.5, 0, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.     , 0.63662, 0.17596, 0.17596, 0.46066, 0.46066]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0, 0.5, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.46066, 0.46066, 0.63662, 0.     , 0.46066, 0.46066]) == 0).all()

    # tests with some MTs present

    test_spindle.remove_microtubules([2, 5])
    
    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.31831, 0.31831, 0.     , 0.31831, 0.31831, 0.     ]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0, 0, 0.5]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.46066, 0.46066, 0.     , 0.17596, 0.     , 0.     ]) == 0).all()

    test_spindle.set_mtoc_pos(np.array([0, -0.5, 0]))
    assert (test_spindle.biased_spatial_catastrophe_distribution().round(5) - np.array([0.46066, 0.46066, 0.     , 0.63662, 0.46066, 0.     ]) == 0).all()

    test_spindle.remove_microtubules([0, 1, 3, 4])


def test_biased_length_nucleation_distribution():

    # tests with no MTs

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0.36788, 0.36788, 0.36788, 0.36788, 0.36788, 0.36788])).all()

    test_spindle.set_mtoc_pos(np.array([0.5, 0, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0.60653, 0.22313, 0.32692, 0.32692, 0.32692, 0.32692])).all()

    test_spindle.set_mtoc_pos(np.array([0, 0.5, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0.32692, 0.32692, 0.60653, 0.22313, 0.32692, 0.32692])).all()

    # test with all MTs present

    test_spindle.add_microtubules([0,1,2,3,4,5])

    test_spindle.set_mtoc_pos(np.array([0, 0.5, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0., 0., 0., 0., 0., 0.])).all()

    # tests with some MTs present 

    test_spindle.remove_microtubules([0, 1, 3, 5])

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0.36788, 0.36788, 0.     , 0.36788, 0.     , 0.36788])).all()

    test_spindle.set_mtoc_pos(np.array([0, -0.5, 0]))
    assert (test_spindle.biased_length_nucleation_distribution().round(5) == np.array([0.32692, 0.32692, 0.     , 0.60653, 0.     , 0.32692])).all()

    test_spindle.remove_microtubules([2, 4])

def test_biased_length_catastrophe_distribution():

    # test with no MTs present
    test_spindle.set_mtoc_pos(np.array([0, -0.5, 0]))
    assert (test_spindle.biased_length_catastrophe_distribution().round(5) == np.array([0., 0., 0., 0., 0., 0.])).all()

    # test with all MTs present
    test_spindle.add_microtubules([0,1,2,3,4,5])

    test_spindle.set_mtoc_pos(np.array([0, 0, 0]))
    assert (test_spindle.biased_length_catastrophe_distribution().round(5) == np.array([0.63212, 0.63212, 0.63212, 0.63212, 0.63212, 0.63212])).all()

    test_spindle.set_mtoc_pos(np.array([0, -0.5, 0]))
    assert (test_spindle.biased_length_catastrophe_distribution().round(5) == np.array([0.67308, 0.67308, 0.77687, 0.39347, 0.67308, 0.67308])).all()

    # tests with some MTs present
    test_spindle.remove_microtubules([1,3,5])

    test_spindle.set_mtoc_pos(np.array([0, -0.5, 0]))
    assert (test_spindle.biased_length_catastrophe_distribution().round(5) == np.array([0.67308, 0.     , 0.77687, 0.     , 0.67308, 0.     ])).all()

    test_spindle.set_mtoc_pos(np.array([0, 0, -0.5]))
    assert (test_spindle.biased_length_catastrophe_distribution().round(5) == np.array([0.67308, 0.     , 0.67308, 0.     , 0.77687, 0.     ])).all()

    # remove microtubules
    test_spindle.remove_microtubules([0,2,4])


def test_spindle_update():
    # testing update spindle

    # given unstable position and some cost

    test_spindle.add_microtubules([1,0])
    test_spindle.set_mtoc_pos(np.array([0, 0.5, 0]))

    old_spindle_state = np.copy(test_spindle.spindle_state)
    old_mtoc_pos = np.copy(test_spindle.mtoc_pos)
    old_cost = test_spindle.calc_cost()

    # call spindle update

    attempts = test_spindle.gradient_descent_spindle_update()

    new_spindle_state = np.copy(test_spindle.spindle_state)
    new_mtoc_pos = np.copy(test_spindle.mtoc_pos)
    new_cost = test_spindle.calc_cost()

    assert (new_spindle_state != old_spindle_state).any()
    assert (new_mtoc_pos != old_mtoc_pos).any()
    assert (new_cost < old_cost)