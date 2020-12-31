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

from dataclasses import dataclass
import dataclasses
import functools
import math
import numpy as np
import scipy.linalg
from scipy.special import logsumexp
import scipy.stats

from . import kalman_filter as kf

@dataclass
class Hyperparameters:
    # The width of the frame in pixels.
    frame_width: float = 640.

    # The height of the frame in pixels.
    frame_height: float = 480.

    # The filter state mean of a ball when it is first added.
    def initial_state_mean(self):
        return np.array([self.frame_width / 2., 0., 0., self.frame_height / 2., 0., 0.])

    # The filter state stdev of ball position when it is first added.
    initial_state_pos_sd: float = 1000.

    # The filter state stdev of ball velocity when it is first addd.
    initial_state_v_sd: float = 1000.

    # The filter state stdev of ball acceleration when it is first addd.
    initial_state_a_sd: float = 1000.

    # The per-second rate at which balls are dropped.
    lambda_drop: float = 1.

    # The per-second rate at which new balls are added.
    lambda_new: float = 1.

    # The probability that a ball is observed in a frame where the ball
    # is present.
    p_obs: float = 0.9

    # The per-frame rate at which spurious observations are made.
    lambda_spur: float = 0.1

    # The parameters for the ball state kalman filter.
    kalman: kf.Hyperparameters = kf.Hyperparameters()

    # A random number generator used for drawing from random
    # distributions.
    rng: np.random.Generator = np.random.default_rng()

    def initial_state_covariance(self):
        """The filter covariance for a ball when it is first added."""
        return np.diag([
            self.initial_state_pos_sd,
            self.initial_state_v_sd,
            self.initial_state_a_sd,
            self.initial_state_pos_sd,
            self.initial_state_v_sd,
            self.initial_state_a_sd,
        ])

    # A cache for `p_counts`.
    p_counts_cache: dict = dataclasses.field(default_factory=dict)

    def p_counts(self, ball_count, measurement_count):
        """Returns:
        - the probability of seeing `measurement_count` given that
          there are `ball_count` balls; and
        - the discrete probability distribution over the number of
          true measurements (measurements actually coming from a ball)
          given that there are `ball_count` balls and
          `measurement_count` measurements.
        """

        cached_result = self.p_counts_cache.get(
            (ball_count, measurement_count))
        if cached_result is not None:
            return cached_result

        true_observation_count_max = min(ball_count, measurement_count)
        result = np.zeros(true_observation_count_max + 1)
        total = 0.0
        for t in range(true_observation_count_max + 1):
            s = measurement_count - t
            p_spur = (self.lambda_spur ** s) * np.exp(-self.lambda_spur) / math.factorial(s)
            p_true = math.comb(ball_count, t) * \
                (self.p_obs ** t) * ((1 - self.p_obs) ** (ball_count - t))
            p = p_spur * p_true
            result[t] = p
            total += p
        result = result / total

        self.p_counts_cache[(ball_count, measurement_count)] = (total, result)
        return (total, result)


@dataclass
class Particle:
    """The joint distribution of ball states given measurements and
    certain discrete hidden variables.

    The discrete hidden variables are:
    - associations between measurements and balls
    - when a new ball gets added
    - when a ball gets dropped.
    """

    filter: kf.States = kf.States()

    def transitioned(self, dt: float, hp: Hyperparameters):
        """Returns a draw from the distribution of `self` transitioned
        forwards by `dt`.

        1. Transitions the current balls using kalman.
        2. Drops balls according to the ball drop distribution.
        3. Adds new balls according to the new ball distribution.
        """
        result = dataclasses.replace(
            self, filter=self.filter.transitioned(dt, hp.kalman))

        # TODO: proper dt-invariance with Poisson and Exponential distributions
        add_ball = hp.rng.uniform() < hp.lambda_new * dt
        drop_ball = hp.rng.uniform(size=[result.filter.means.shape[0]]) < \
            hp.lambda_drop * dt

        result.filter.means = result.filter.means[~drop_ball, :]
        result.filter.covariances = result.filter.covariances[~drop_ball, :]

        if add_ball:
            result.filter.means = np.concatenate([
                result.filter.means,
                hp.initial_state_mean()[np.newaxis, ...]
            ])
            result.filter.covariances = np.concatenate([
                result.filter.covariances,
                hp.initial_state_covariance()[np.newaxis, ...]
            ])

        return result

    def posterior(self, measurements: np.ndarray, hp: Hyperparameters):
        # TODO: For better testability, refactor this into separate sampling and deterministic
        # steps.
        """Returns:
        - a draw from the posterior distribution given `self` and
          `measurements`; and
        - the log liklihood of `measurements` under the observation model
          given `self`.
        """

        ball_count = self.filter.means.shape[0]
        measurement_count = measurements.shape[0]
        p_measurement_count, p_true_observation_count = \
            hp.p_counts(ball_count, measurement_count)
        true_observation_count = hp.rng.choice(
            p_true_observation_count.shape[0], p=p_true_observation_count)

        observed_ball_indices = hp.rng.choice(
            ball_count, size=true_observation_count, replace=False)
        true_observation_indices = hp.rng.choice(
            measurement_count, size=true_observation_count, replace=False)

        ball_mask = np.full(ball_count, False)
        ball_mask[observed_ball_indices] = True
        obs = measurements[true_observation_indices, :]

        updated_filter, kalman_logp = self.filter.posterior(
            ball_mask, obs, hp.kalman)

        # Liklihood of seeing the spurious observations where they are.
        spurious_logp = (measurement_count - true_observation_count) * \
            -np.log(hp.frame_width * hp.frame_height)

        return (
            Particle(filter=updated_filter),
            kalman_logp + spurious_logp + np.log(p_measurement_count)
        )


@dataclass
class State:
    particles: list # list<Particle>
