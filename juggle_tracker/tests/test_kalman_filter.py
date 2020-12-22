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

import numpy as np
import scipy.linalg

import juggle_tracker.kalman_filter as kf

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

    new_states = kf.kalman_transition(states, 1., hp)

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
