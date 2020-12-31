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

import pytest
import math
import numpy as np
import scipy.linalg

import juggle_tracker.kalman_filter as kf
import juggle_tracker.kf_particle as kp

from . import assertions
from . import util


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
                hp.initial_state_mean())
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
            means=np.tile(hp.initial_state_mean()[np.newaxis, :], (3, 1)),
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


def check_p_counts(
        hp,
        ball_count,
        measurement_count,
        expected_p_measurement_count,
        expected_p_true_observation_count):
    p_measurement_count, p_true_observation_count = hp.p_counts(
        ball_count, measurement_count)
    assert(p_measurement_count == pytest.approx(expected_p_measurement_count))
    np.testing.assert_allclose(
        p_true_observation_count, expected_p_true_observation_count)


def test_p_counts():
    no_spurious = kp.Hyperparameters(p_obs=0.9, lambda_spur=0.)
    check_p_counts(
        no_spurious, 0, 0,
        expected_p_measurement_count=1.,
        expected_p_true_observation_count=[1.])
    check_p_counts(
        no_spurious, 5, 0,
        expected_p_measurement_count=0.1 ** 5,
        expected_p_true_observation_count=[1.])
    check_p_counts(
        no_spurious, 5, 3,
        expected_p_measurement_count = 10 * (0.9 ** 3) * (0.1 ** 2),
        expected_p_true_observation_count = [0., 0., 0., 1.])
    check_p_counts(
        no_spurious, 5, 5,
        expected_p_measurement_count = 0.9 ** 5,
        expected_p_true_observation_count = [0., 0., 0., 0., 0., 1.])

    all_spurious = kp.Hyperparameters(p_obs=0., lambda_spur=0.1)
    check_p_counts(
        all_spurious, 0, 0,
        expected_p_measurement_count = np.exp(-0.1),
        expected_p_true_observation_count = [1.])
    check_p_counts(
        all_spurious, 0, 2,
        expected_p_measurement_count = (0.1 ** 2) * np.exp(-0.1) / 2,
        expected_p_true_observation_count = [1.])
    check_p_counts(
        all_spurious, 5, 0,
        expected_p_measurement_count = np.exp(-0.1),
        expected_p_true_observation_count = [1.])
    check_p_counts(
        all_spurious, 5, 3,
        expected_p_measurement_count = (0.1 ** 3) * np.exp(-0.1) / 6,
        expected_p_true_observation_count = [1., 0., 0., 0.])
    check_p_counts(
        all_spurious, 5, 5,
        expected_p_measurement_count = (0.1 ** 5) * np.exp(-0.1) / math.factorial(5),
        expected_p_true_observation_count = [1., 0., 0., 0., 0., 0.])
    check_p_counts(
        all_spurious, 5, 7,
        expected_p_measurement_count = (0.1 ** 7) * np.exp(-0.1) / math.factorial(7),
        expected_p_true_observation_count = [1., 0., 0., 0., 0., 0.])

    some_spurious = kp.Hyperparameters(p_obs=0.9, lambda_spur=0.1)
    check_p_counts(
        some_spurious, 0, 0,
        expected_p_measurement_count = np.exp(-0.1),
        expected_p_true_observation_count = [1.])
    check_p_counts(
        some_spurious, 0, 2,
        expected_p_measurement_count = (0.1 ** 2) * np.exp(-0.1) / 2,
        expected_p_true_observation_count = [1.])
    check_p_counts(
        some_spurious, 1, 0,
        expected_p_measurement_count = np.exp(-0.1) * 0.1,
        expected_p_true_observation_count = [1.])
    check_p_counts(
        some_spurious, 1, 1,
        expected_p_measurement_count = np.exp(-0.1) * 0.9 + 0.1 * np.exp(-0.1) * 0.1,
        expected_p_true_observation_count = [0.01 / 0.91, 0.9 / 0.91])


def test_posterior_no_balls_no_measurements():
    hp = kp.Hyperparameters(
        p_obs=0.9,
        lambda_spur=0.1,
        rng=np.random.default_rng(5))
    particle = kp.Particle()
    updated_particle, logp = particle.posterior(np.array(np.zeros((0, 2))), hp)
    np.testing.assert_allclose(updated_particle.filter.means, np.zeros((0, 6)))
    np.testing.assert_allclose(updated_particle.filter.covariances, np.zeros((0, 6, 6)))
    assert(logp == pytest.approx(-0.1))


def test_posterior_no_balls():
    hp = kp.Hyperparameters(
        frame_width=640., frame_height=480.,
        p_obs=0.9,
        lambda_spur=0.1,
        rng=np.random.default_rng(5))
    particle = kp.Particle()
    updated_particle, logp = particle.posterior(np.array([[320., 240.]]), hp)
    np.testing.assert_allclose(updated_particle.filter.means, np.zeros((0, 6)))
    np.testing.assert_allclose(updated_particle.filter.covariances, np.zeros((0, 6, 6)))

    # Likelihood of one spurious measurement.
    expected_logp_poisson = np.log(0.1) - 0.1

    # Liklihood of the spurious measurement being where it is.
    expected_logp_uniform = -np.log(640. * 480.)

    expected_logp = expected_logp_poisson + expected_logp_uniform

    assert(logp == pytest.approx(expected_logp))


def test_posterior_one_ball_measured():
    hp = kp.Hyperparameters(
        frame_width=640., frame_height=480.,
        p_obs=0.8,
        lambda_spur=1.,
        rng=np.random.default_rng(5))
    particle = kp.Particle(
        filter=kf.States(
            means=np.array([[320., 0., 0., 240., 0., 0.]]),
            covariances=hp.initial_state_covariance()[np.newaxis, ...]))

    # We expect one of two things to happen:
    # The measurement is for the ball, with probability 0.8
    # The measurement is spurious, with probability 0.2

    pass


def test_posterior_one_ball_no_measurement():
    pass


def test_posterior_one_ball_spurious_measurement():
    pass


def test_posterior_one_ball_measured_and_spurious_measurement():
    pass


def test_posterior_many_balls_many_measurements():
    pass
