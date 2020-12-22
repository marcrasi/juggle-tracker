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

from dataclasses import dataclass, field, replace
import functools
import math
import numpy as np
import scipy.linalg
from scipy.special import logsumexp
import scipy.stats

@dataclass
class KalmanHyperparameters:
    transition_pos_sd: float = 10.
    transition_v_sd: float = 10.
    transition_a_sd: float = 10_000.
    observation_sd: float = 20.

@dataclass
class KalmanStates:
    # The ball state vector is [x, vx, ax, y, vy, ay].

    # [ball_count, 6] array of KF means for ball states.
    means: np.ndarray

    # [ball_count, 6, 6] array of KF covariances for ball states.
    covariances: np.ndarray

def transition_matrix(dt: float):
    a = np.array([[1, dt, 0.5 * (dt**2)],
                  [0, 1,            dt],
                  [0, 0,            1]])
    return scipy.linalg.block_diag(a, a)

def kalman_transition(states: KalmanStates, dt: float, hp: KalmanHyperparameters):
    t = transition_matrix(dt)[np.newaxis, ...]
    tt = np.transpose(t, axes=(0, 2, 1))
    # TODO: Multiply sds by sqrt(dt)??
    transition_pos_sd = hp.transition_pos_sd
    transition_v_sd = hp.transition_v_sd
    transition_a_sd = hp.transition_a_sd

    # TODO: Make this stuff a hyperparameter.
    # idea!!! I can just look at the empirical distribution of the acceleration
    # changes (estimated by a simple smoother with a big SD, maybe) and then
    # fit a mixture of gaussians to that!!
    # transition_a_sd = hp.transition_a_sd if np.random.uniform() < 0.8 else 10
    
    process_noise_cov = np.diag([
        transition_pos_sd**2, transition_v_sd**2, transition_a_sd**2,
        transition_pos_sd**2, transition_v_sd**2, transition_a_sd**2,
    ])[np.newaxis, ...]
    return KalmanStates(
        means = (t @ states.means[..., np.newaxis]).squeeze(-1),
        covariances = t @ states.covariances @ tt + process_noise_cov,
    )