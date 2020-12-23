
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
    initial_state_mean: np.ndarray = np.array([240., 0., 0., 320., 0., 0.])
    initial_state_pos_sd: float = 10.
    initial_state_v_sd: float = 1000.
    initial_state_a_sd: float = 1000.

    lambda_drop: float = 1.
    lambda_new: float = 1.
    p_obs: float = 0.9
    lambda_spur: float = 0.1

    kalman: kf.Hyperparameters = kf.Hyperparameters()

    rng: np.random.Generator = np.random.default_rng()

    def initial_state_covariance(self):
        return np.diag([
            self.initial_state_pos_sd,
            self.initial_state_v_sd,
            self.initial_state_a_sd,
            self.initial_state_pos_sd,
            self.initial_state_v_sd,
            self.initial_state_a_sd,
        ])


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
        """Returns `self`, transitioned forwards in time by `dt`.

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
                hp.initial_state_mean[np.newaxis, ...]
            ])
            result.filter.covariances = np.concatenate([
                result.filter.covariances,
                hp.initial_state_covariance()[np.newaxis, ...]
            ])

        return result

    def posterior(self, measurements: np.ndarray, hp: Hyperparameters):
        pass

@dataclass
class State:
    particles: list # list<Particle>
