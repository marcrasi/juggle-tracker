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
class Particle:
    # The ball state vector is [x, vx, ax, y, vy, ay]

    # [ball_count, 6] array of KF means for ball positions
    means: np.ndarray

    # [ball_count, 6, 6] array of KF covariances for ball positions
    covariances: np.ndarray

    # [history_size] circular buffer of ancestor particle ids
    ancestors: list

    ancestor_buffer_cur: int

    def id(self):
        return self.ancestors[self.ancestor_buffer_cur]

    def oldest_ancestor(self):
        return self.ancestors[(self.ancestor_buffer_cur + 1) % self.ancestors.shape[0]]

@dataclass
class Hyperparameters:
    initial_state_mean: np.ndarray = np.array([240, 0, 0, 320, 0, 0])
    initial_state_sd: float = 1_000
    transition_pos_sd: float = 10
    transition_v_sd: float = 10
    transition_a_sd: float = 10_000
    observation_sd: float = 20

    lambda_drop: float = 1.0
    lambda_new: float = 1.0
    p_obs: float = 0.9
    lambda_spur: float = 0.1

    max_particles: int = 1000

    p_true_observation_count_cache: dict = field(default_factory=dict)

    def p_true_observation_count(self, ball_count, measurement_count):
        cached_result = self.p_true_observation_count_cache.get((ball_count, measurement_count))
        if cached_result is not None:
            return cached_result

        true_observation_count_max = min(ball_count, measurement_count)
        result = np.zeros(true_observation_count_max + 1)
        total = 0.0
        for t in range(true_observation_count_max + 1):
            s = measurement_count - t
            p_spur = (self.lambda_spur ** s) * np.exp(-self.lambda_spur) / math.factorial(s)
            p_true = math.comb(ball_count, t) * (self.p_obs ** t) * ((1 - self.p_obs) ** s)
            p = p_spur * p_true
            result[t] = p
            total += p
        result = result / total

        self.p_true_observation_count_cache[(ball_count, measurement_count)] = result
        return result

def transition_matrix(dt: float):
    a = np.array([[1, dt, 0.5 * (dt**2)],
                  [0, 1,            dt],
                  [0, 0,            1]])
    return scipy.linalg.block_diag(a, a)

def kalman_transition(particle: Particle, dt: float, hp: Hyperparameters):
    t = transition_matrix(dt)[np.newaxis, ...]
    tt = np.transpose(t, axes=(0, 2, 1))
    # TODO: Multiply sds by sqrt(dt)??
    transition_pos_sd = hp.transition_pos_sd
    transition_v_sd = hp.transition_v_sd
    # idea!!! I can just look at the empirical distribution of the acceleration
    # changes (estimated by a simple smoother with a big SD, maybe) and then
    # fit a mixture of gaussians to that!!
    transition_a_sd = hp.transition_a_sd if np.random.uniform() < 0.8 else 10
    process_noise_cov = np.diag([
        transition_pos_sd**2, transition_v_sd**2, transition_a_sd**2,
        transition_pos_sd**2, transition_v_sd**2, transition_a_sd**2,
    ])[np.newaxis, ...]
    return replace(
        particle,
        means = (t @ particle.means[..., np.newaxis]).squeeze(-1),
        covariances = t @ particle.covariances @ tt + process_noise_cov,
    )

# ball_mask is [ball_count] boolean array saying whether we have an observation
# for the corresponding ball
# obs is [sum(ball_mask), 2] array of (x, y) observations
def kalman_posterior(particle: Particle, ball_mask: np.ndarray, obs: np.ndarray, hp: Hyperparameters):
    if np.sum(ball_mask) == 0:
        return particle, 0.0

    observation_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ])[np.newaxis, ...]
    observation_matrix_T = np.transpose(observation_matrix, axes=(0, 2, 1))
    obs_noise_cov = np.diag([hp.observation_sd**2, hp.observation_sd**2])[np.newaxis, ...]

    masked_means = particle.means[ball_mask, ..., np.newaxis]
    masked_covs = particle.covariances[ball_mask, ...]

    residual = obs[..., np.newaxis] - (observation_matrix @ masked_means)
    residual_T = np.transpose(residual, axes=(0, 2, 1))
    S = observation_matrix @ masked_covs @ observation_matrix_T + obs_noise_cov
    S_inv = np.linalg.inv(S)
    K = masked_covs @ observation_matrix_T @ S_inv

    n_obs = 2 * np.sum(ball_mask).item()
    a = -0.5 * (residual_T @ S_inv @ residual).sum().item()
    b = -0.5 * n_obs * np.log(2 * math.pi).item()
    c = -0.5 * np.log(np.linalg.det(S)).sum().item()
    obs_logp = a + b + c

    new_means = particle.means.copy()
    new_covs = particle.covariances.copy()
    new_means[ball_mask, ...] = (masked_means + (K @ residual)).squeeze(-1)
    new_covs[ball_mask, ...] = (np.eye(6) - K @ observation_matrix) @ masked_covs

    return replace(particle, means=new_means, covariances=new_covs), obs_logp

def step(particle: Particle, dt: float, hp: Hyperparameters, next_particle_id: int):
    result = kalman_transition(particle, dt, hp)

    result.ancestor_buffer_cur += 1
    result.ancestor_buffer_cur %= result.ancestors.shape[0]
    result.ancestors = result.ancestors.copy()
    result.ancestors[result.ancestor_buffer_cur] = next_particle_id

    # TODO: proper dt-invariance with Poisson and Exponential distributions
    add_ball = np.random.uniform() < hp.lambda_new * dt
    drop_ball = np.random.uniform(size=[result.means.shape[0]]) < hp.lambda_drop * dt

    result.means = result.means[~drop_ball, :]
    result.covariances = result.covariances[~drop_ball, :]

    if add_ball:
        result.means = np.concatenate([
            result.means,
            hp.initial_state_mean[np.newaxis, ...]
        ])
        result.covariances = np.concatenate([
            result.covariances,
            np.diag([hp.initial_state_sd**2] * 6)[np.newaxis, ...]
        ])

    return result

def observe(particle: Particle, measurements: np.ndarray, hp: Hyperparameters):
    ball_count = particle.means.shape[0]
    measurement_count = measurements.shape[0]
    p_true_observation_count = hp.p_true_observation_count(ball_count, measurement_count)
    true_observation_count = np.random.choice(
        p_true_observation_count.shape[0],
        p=p_true_observation_count
    )

    # if measurement_count > 0:
    #     print(true_observation_count, measurement_count)

    observed_ball_indices = np.random.choice(
        ball_count,
        size=true_observation_count,
        replace=False
    )
    true_observation_indices = np.random.choice(
        measurement_count,
        size=true_observation_count,
        replace=False
    )

    ball_mask = np.full(ball_count, False)
    ball_mask[observed_ball_indices] = True
    obs = measurements[true_observation_indices, :]

    # if np.sum(ball_mask) > 0:
    #     print(ball_mask, true_observation_indices)
    updated_particle, kalman_logp = kalman_posterior(particle, ball_mask, obs, hp)
    spurious_logp = (measurement_count - true_observation_count) * (-np.log(640 * 480))

    return updated_particle, kalman_logp + spurious_logp

def particle_filter_update(
    particles: list, dt: float, measurements: np.ndarray, hp: Hyperparameters,
    next_particle_id: int
):
    stepped_particles = []
    for p in particles:
        stepped_particles.append(step(p, dt, hp, next_particle_id))
        next_particle_id += 1

    observed_particles = [observe(p, measurements, hp) for p in stepped_particles]
    weights = np.array([p[1] for p in observed_particles])
    weights -= logsumexp(weights)

    # So the rest of this finds the ancestor (of the depth of the buffer) with
    # the most probability, filters to keep just the particles with that
    # ancestor, and then resamples according to the remaining probabilities.
    # TODO: There is surely a simpler cleaner way to do this!!

    ps = np.exp(weights)
    oldest_ancestor_ps = {}
    most_probable_old_ancestor = None
    for i in range(len(observed_particles)):
        p = observed_particles[i][0]
        if p.oldest_ancestor() not in oldest_ancestor_ps:
            oldest_ancestor_ps[p.oldest_ancestor()] = ps[i]
        else:
            oldest_ancestor_ps[p.oldest_ancestor()] += ps[i]
        if most_probable_old_ancestor is None or oldest_ancestor_ps[p.oldest_ancestor()] > oldest_ancestor_ps[most_probable_old_ancestor]:
            most_probable_old_ancestor = p.oldest_ancestor()

    if most_probable_old_ancestor is not None:
        observed_particles = [p for p in observed_particles if p[0].oldest_ancestor() == most_probable_old_ancestor]
        weights = np.array([p[1] for p in observed_particles])
        weights -= logsumexp(weights)

    resampled_indices = np.random.choice(len(observed_particles), size=hp.max_particles, p=np.exp(weights))
    return [observed_particles[i][0] for i in resampled_indices], most_probable_old_ancestor
