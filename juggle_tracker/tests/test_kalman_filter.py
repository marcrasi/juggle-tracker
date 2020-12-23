# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import scipy.linalg

import juggle_tracker.kalman_filter as kf

def normal_log_pdf(x, mean, variance):
    return -0.5 * np.log(2 * math.pi * variance) + \
            -0.5 * ((x - mean) ** 2) / variance 

def test_kalman_transition():
    hp = kf.KalmanHyperparameters(
        transition_pos_sd = 1.,
        transition_v_sd = 2.,
        transition_a_sd = 3.,
        observation_sd = 20.,
    )
    states = kf.KalmanStates(
        means = np.array([
            [0., 10., 0., 0., 20., 10.],
            [0., -5., 0., 0., -10., 0.],
        ]),
        covariances = np.array([
            np.diag([1., 1., 1., 1., 1., 1.]),
            np.diag([1., 1., 1., 1., 1., 1.]),
        ])
    )

    new_states = states.transitioned(1., hp)

    expected_means = np.array([
        [10., 10., 0., 25., 30., 10.],
        [-5., -5., 0., -10., -10., 0.],
    ])
    np.testing.assert_allclose(new_states.means, expected_means)
    expected_cov_block = np.array([
        [3.25, 1.5, 0.5],
        [1.5, 6., 1.],
        [0.5, 1., 10.],
    ])
    expected_covariances = np.array([
        scipy.linalg.block_diag(expected_cov_block, expected_cov_block),
        scipy.linalg.block_diag(expected_cov_block, expected_cov_block),
    ])
    np.testing.assert_allclose(new_states.covariances, expected_covariances)

def test_kalman_posterior_at_mean():
    """One ball, one observation exactly at the mean of the state."""
    hp = kf.KalmanHyperparameters(
        transition_pos_sd = 1.,
        transition_v_sd = 2.,
        transition_a_sd = 3.,
        observation_sd = 2.,
    )
    states = kf.KalmanStates(
        means = np.array([
            [0., 10., 0., 0., 20., 10.],
        ]),
        covariances = np.array([
            np.diag([1., 1., 1., 1., 1., 1.]),
        ])
    )

    new_states, observation_logp = states.posterior(
        np.array([True]), np.array([0., 0.]), hp)

    expected_means = states.means
    np.testing.assert_allclose(new_states.means, expected_means)
    expected_cov_block = np.array([
        [0.8, 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    expected_covariances = np.array([
        scipy.linalg.block_diag(expected_cov_block, expected_cov_block)
    ])
    np.testing.assert_allclose(new_states.covariances, expected_covariances)

    # The x-coordinate observation has mean 0 and variance
    #   state_variance + observation_variance = 1 + 4 = 5.
    # The y-coordinate observation has the same mean and variance.
    expected_observation_logp = normal_log_pdf(0., mean=0., variance=5.) + \
        normal_log_pdf(0., mean=0., variance=5.)
    np.testing.assert_allclose(observation_logp, expected_observation_logp)

def test_kalman_posterior_away_mean():
    """One ball, one observation away from the mean of the state."""
    hp = kf.KalmanHyperparameters(
        transition_pos_sd = 1.,
        transition_v_sd = 2.,
        transition_a_sd = 3.,
        observation_sd = 2.,
    )
    states = kf.KalmanStates(
        means = np.array([
            [0., 10., 0., 0., 20., 10.],
        ]),
        covariances = np.array([
            np.diag([1., 1., 1., 1., 1., 1.]),
        ])
    )

    new_states, observation_logp = states.posterior(
        np.array([True]), np.array([10., 20.]), hp)

    expected_means = np.array([[2., 10., 0., 4., 20., 10.]])
    np.testing.assert_allclose(new_states.means, expected_means)
    expected_cov_block = np.array([
        [0.8, 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    expected_covariances = np.array([
        scipy.linalg.block_diag(expected_cov_block, expected_cov_block)
    ])
    np.testing.assert_allclose(new_states.covariances, expected_covariances)

    # The x-coordinate observation has mean 0 and variance
    #   state_variance + observation_variance = 1 + 4 = 5.
    # The y-coordinate observation has the same mean and variance.
    expected_observation_x_logp = -0.5 * np.log(2 * math.pi * 5) - 0.5 * (10. / math.sqrt(5.)) ** 2
    expected_observation_y_logp = -0.5 * np.log(2 * math.pi * 5) - 0.5 * (20. / math.sqrt(5.)) ** 2
    expected_observation_logp = normal_log_pdf(10., mean=0., variance=5.) + \
        normal_log_pdf(20., mean=0., variance=5.)
    np.testing.assert_allclose(observation_logp, expected_observation_logp)

def test_kalman_posterior_mask():
    """Two balls, but one of them is not observed."""
    hp = kf.KalmanHyperparameters(
        transition_pos_sd = 1.,
        transition_v_sd = 2.,
        transition_a_sd = 3.,
        observation_sd = 2.,
    )
    states = kf.KalmanStates(
        means = np.array([
            [15., 10., 0., 30., 20., 10.],
            [0., 10., 0., 0., 20., 10.],
        ]),
        covariances = np.array([
            np.diag([1., 1., 1., 1., 1., 1.]),
            np.diag([1., 1., 1., 1., 1., 1.]),
        ])
    )

    new_states, observation_logp = states.posterior(
        np.array([False, True]), np.array([0., 0.]), hp)

    expected_means = states.means
    np.testing.assert_allclose(new_states.means, expected_means)
    expected_cov_block = np.array([
        [0.8, 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    expected_covariances = np.array([
        np.diag([1., 1., 1., 1., 1., 1.]),
        scipy.linalg.block_diag(expected_cov_block, expected_cov_block),
    ])
    np.testing.assert_allclose(new_states.covariances, expected_covariances)

    # The x-coordinate observation has mean 0 and variance
    #   state_variance + observation_variance = 1 + 4 = 5.
    # The y-coordinate observation has the same mean and variance.
    expected_observation_logp = normal_log_pdf(0., mean=0., variance=5.) + \
        normal_log_pdf(0., mean=0., variance=5.)
    np.testing.assert_allclose(observation_logp, expected_observation_logp)
