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
import juggle_tracker.kf_particle as kp

def test_add_ball():
    """Test that we add a ball to an empty particle with the correct
    probability."""

    hp = kp.Hyperparameters(
        lambda_new = 0.2,
        rng = np.random.default_rng(42))
    particle = kp.Particle()

    n_added = 0
    for _ in range(100):
        new_particle = particle.transitioned(0.5, hp)
        assert(new_particle.filter.means.shape[0] == \
                new_particle.filter.covariances.shape[0])
        assert(new_particle.filter.means.shape[0] in [0, 1])
        assert(new_particle.filter.covariances.shape[0] in [0, 1])
        if new_particle.filter.means.shape[0] == 1:
            np.testing.assert_allclose(
                new_particle.filter.means[0],
                hp.initial_state_mean)
            np.testing.assert_allclose(
                new_particle.filter.covariances[0],
                hp.initial_state_covariance())
            n_added += 1

    assert(n_added == 9)

def test_remove_ball():
    """Test that we remove balls with the correct probability from a
    particle with 3 balls."""

    hp = kp.Hyperparameters(
        lambda_drop = 0.2,
        lambda_new = 0.,
        rng = np.random.default_rng(42))
    particle = kp.Particle(
        filter=kf.States(
            means=np.tile(hp.initial_state_mean[np.newaxis, :], (3, 1)),
            covariances=np.tile(
                hp.initial_state_covariance()[np.newaxis, :], (3, 1, 1))))

    n_removed = 0
    for _ in range(100):
        new_particle = particle.transitioned(0.5, hp)
        assert(new_particle.filter.means.shape[0] == \
                new_particle.filter.covariances.shape[0])
        assert(new_particle.filter.means.shape[0] in [0, 1, 2, 3])
        assert(new_particle.filter.covariances.shape[0] in [0, 1, 2, 3])
        n_removed += 3 - new_particle.filter.means.shape[0]

    assert(n_removed == 25)
