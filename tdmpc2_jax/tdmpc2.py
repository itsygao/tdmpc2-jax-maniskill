from __future__ import annotations
from functools import partial
from flax import struct
import jax
from jaxtyping import PRNGKeyArray
import optax

from tdmpc2_jax.world_model import WorldModel
import jax.numpy as jnp
from tdmpc2_jax.common.loss import soft_crossentropy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from tdmpc2_jax.common.scale import percentile_normalization
from tdmpc2_jax.common.util import sg


class TDMPC2(struct.PyTreeNode):
  model: WorldModel
  scale: jax.Array

  # Planning
  mpc: bool
  horizon: int = struct.field(pytree_node=False)
  mppi_iterations: int = struct.field(pytree_node=False)
  population_size: int = struct.field(pytree_node=False)
  policy_prior_samples: int = struct.field(pytree_node=False)
  num_elites: int = struct.field(pytree_node=False)
  min_plan_std: float
  max_plan_std: float
  temperature: float

  # Optimization
  batch_size: int = struct.field(pytree_node=False)
  discount: float
  rho: float
  consistency_coef: float
  reward_coef: float
  value_coef: float
  continue_coef: float
  entropy_coef: float
  tau: float

  @classmethod
  def create(cls,
             world_model: WorldModel,
             # Planning
             mpc: bool,
             horizon: int,
             mppi_iterations: int,
             population_size: int,
             policy_prior_samples: int,
             num_elites: int,
             min_plan_std: float,
             max_plan_std: float,
             temperature: float,
             # Optimization
             discount: float,
             batch_size: int,
             rho: float,
             consistency_coef: float,
             reward_coef: float,
             value_coef: float,
             continue_coef: float,
             entropy_coef: float,
             tau: float
             ) -> TDMPC2:

    return cls(model=world_model,
               mpc=mpc,
               horizon=horizon,
               mppi_iterations=mppi_iterations,
               population_size=population_size,
               policy_prior_samples=policy_prior_samples,
               num_elites=num_elites,
               min_plan_std=min_plan_std,
               max_plan_std=max_plan_std,
               temperature=temperature,
               discount=discount,
               batch_size=batch_size,
               rho=rho,
               consistency_coef=consistency_coef,
               reward_coef=reward_coef,
               value_coef=value_coef,
               continue_coef=continue_coef,
               entropy_coef=entropy_coef,
               tau=tau,
               scale=jnp.array([1.0]),
               )

  def act(self,
          obs: np.ndarray,
          prev_plan: Optional[Tuple[jax.Array]] = None,
          train: bool = True,
          *,
          key: PRNGKeyArray):
    z = self.model.encode(obs, self.model.encoder.params)

    if self.mpc:

      num_envs = z.shape[0] if z.ndim > 1 else 1
      if prev_plan is None:
        prev_plan = (
            jnp.zeros((num_envs, self.horizon, self.model.action_dim)),
            jnp.full((num_envs, self.horizon, self.model.action_dim),
                     self.max_plan_std)
        )
      action, plan = self.plan(
          jnp.atleast_2d(z), prev_plan, train, jax.random.split(key, num_envs))
      action = action.squeeze(0) if z.ndim == 1 else action

    else:
      action = self.model.sample_actions(
          z, self.model.policy_model.params, key=key)[0]
      plan = None

    return np.array(action), plan

  @jax.jit
  @partial(jax.vmap, in_axes=(None, 0, 0, None, 0), out_axes=0)
  def plan(self,
           z: jax.Array,
           prev_plan: Tuple[jax.Array, jax.Array], 
           train: bool,
           key: PRNGKeyArray,
           ) -> Tuple[jax.Array, jax.Array]:
    """
    Select next action via MPPI planner

    Parameters
    ----------
    z : jax.Array
        Enncoded environment observation
    key : PRNGKeyArray
        Jax PRNGKey
    prev_mean : jax.Array, optional
        Mean from previous planning interval. If present, MPPI is given a warm start by time-shifting this value by 1 step. If None, the MPPI mean is set to zero, by default None
    train : bool, optional
        If True, inject noise into the final selected action, by default False

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        - Action output from planning
        - Final mean value (for use in warm start)
    """
    z = jnp.atleast_2d(z) # before vmap, (1, latent_dim) guaranteed 2d
    # prev_plan: before vmap, tuple of (horizon, action_dim)
    # Sample trajectories from policy prior
    key, *prior_keys = jax.random.split(key, self.horizon + 1)
    policy_actions = jnp.zeros(
        (self.horizon, self.policy_prior_samples, self.model.action_dim))
    _z = z.repeat(self.policy_prior_samples, axis=0) # (policy_prior_samples, latent_dim)
    for t in range(self.horizon-1):
      policy_actions = policy_actions.at[t].set(
          self.model.sample_actions(_z, self.model.policy_model.params, key=prior_keys[t])[0])
      _z = self.model.next(
          _z, policy_actions[t], self.model.dynamics_model.params)
    policy_actions = policy_actions.at[-1].set(
        self.model.sample_actions(_z, self.model.policy_model.params, key=prior_keys[-1])[0])

    # Initialize population state
    z = z.repeat(self.population_size, axis=0) # (population_size=num_samples, latent_dim)
    mean = jnp.zeros((self.horizon, self.model.action_dim))
    std = jnp.full((self.horizon, self.model.action_dim), self.max_plan_std)
    # Warm start MPPI with the previous solution
    mean = mean.at[:-1].set(prev_plan[0][1:])
    # std = std.at[:-1].set(prev_plan[1][1:])

    actions = jnp.zeros(
        (self.horizon, self.population_size, self.model.action_dim))
    actions = actions.at[:, :self.policy_prior_samples].set(policy_actions)

    # Iterate MPPI
    key, action_noise_key, *value_keys = \
        jax.random.split(key, self.mppi_iterations+1+1)
    noise = jax.random.normal(
        action_noise_key,
        shape=(
            self.mppi_iterations,
            self.horizon,
            self.population_size - self.policy_prior_samples,
            self.model.action_dim
        ))
    for i in range(self.mppi_iterations):
      # Sample actions
      actions = actions.at[:, self.policy_prior_samples:].set(
          mean[:, None, :] + std[:, None, :] * noise[i])
      actions = actions.clip(-1, 1)

      # Compute elite actions
      value = self.estimate_value(z, actions, key=value_keys[i])
      value = jnp.nan_to_num(value)
      _, elite_inds = jax.lax.top_k(value, self.num_elites)
      elite_values, elite_actions = value[elite_inds], actions[:, elite_inds]

      # Update parameters
      max_value = jnp.max(elite_values)
      score = jnp.exp(self.temperature * (elite_values - max_value))
      score /= score.sum(0)

      mean = jnp.sum(score[None, :, None] * elite_actions, axis=1) / \
          (score.sum(0) + 1e-9)
      std = jnp.sqrt(
          jnp.sum(score[None, :, None] * (elite_actions -
                  mean[:, None, :])**2, axis=1) / (score.sum(0) + 1e-9)
      ).clip(self.min_plan_std, self.max_plan_std)

    # Select action based on the score
    key, *final_action_keys = jax.random.split(key, 3)
    action_ind = jax.random.choice(final_action_keys[0],
                                   a=jnp.arange(self.num_elites), p=score)
    actions = elite_actions[:, action_ind]

    action, action_std = actions[0], std[0]
    action += jnp.array(train, float) * action_std * jax.random.normal(
        final_action_keys[1], shape=action.shape)

    action = action.clip(-1, 1)
    return action, (mean, std)

  @jax.jit
  def update(self,
             observations: jax.Array,                  # observations (horizon, batch_sz, obs_dim)
             actions: jax.Array,                       # actions (horizon, batch_sz)
             rewards: jax.Array,                       # rewards (horizon, batch_sz)
             next_observations: jax.Array,             # next_observations (horizon, batch_sz, obs_dim)
             terminated: jax.Array,                    # terminated (horizon, batch_sz)
             truncated: jax.Array,                     # truncated (horizon, batch_sz)
             *,
             key: PRNGKeyArray
             ) -> Tuple[TDMPC2, Dict[str, Any]]:

    world_model_key, policy_key = jax.random.split(key, 2)

    def world_model_loss_fn(encoder_params: Dict,
                            dynamics_params: Dict,
                            value_params: Dict,
                            reward_params: Dict,
                            continue_params: Dict):
      target_key, Q_key = jax.random.split(world_model_key, 2)
      done = jnp.logical_or(terminated, truncated)
      finished = jnp.zeros((self.horizon+1, self.batch_size), dtype=bool)        # finished (horizon+1, batch_size)
                                                                                 # if finished[i], then obs[i] -> a[i] x->x next_obs[i] don't work

      next_z = sg(self.model.encode(next_observations, encoder_params))          # next_z (horizon, batch_sz, latent_dim)
    #   next_z_grad = self.model.encode(observations, encoder_params)
    #   next_z = sg(self.model.encode(next_observations, encoder_params))
      td_targets = self.td_target(next_z, rewards, terminated, key=target_key)   # td_targets (horizon, batch_sz)

      # Latent rollout (compute latent dynamics + consistency loss)
      zs = jnp.zeros((self.horizon+1, self.batch_size, next_z.shape[-1]))        # zs (horizon+1, batch_size, latent_dim)
      z = self.model.encode(jax.tree.map(
          lambda x: x[0], observations), encoder_params)                         # z (horizon, batch_sz, latent_dim)
      zs = zs.at[0].set(z)
      consistency_loss = 0
      for t in range(self.horizon):                                              # filling in zs with imagined z
        z = self.model.next(z, actions[t], dynamics_params)
        zs = zs.at[t+1].set(z)
        consistency_loss += self.rho**t * \
            jnp.mean((z - next_z[t])**2, where=~finished[t][:, None])

        # Keep track of which trajectories have reached a terminal state
        finished = finished.at[t+1].set(jnp.logical_or(finished[t], done[t]))    # last finished[-1] do what? seems unused later.

      # Get logits for loss computations
      _, q_logits = self.model.Q(zs[:-1], actions, value_params, key=Q_key)
      _, reward_logits = self.model.reward(zs[:-1], actions, reward_params)
    #   _, q_logits = self.model.Q(next_z_grad, actions, value_params, key=Q_key)
    #   _, reward_logits = self.model.reward(next_z_grad, actions, reward_params)
      if self.model.predict_continues:                                           # ?
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, zs[1:]).squeeze(-1)

      reward_loss = 0
      value_loss = 0
      for t in range(self.horizon):
        reward_loss += self.rho**t * soft_crossentropy(
            reward_logits[t], rewards[t],
            self.model.symlog_min,
            self.model.symlog_max,
            self.model.num_bins).mean(where=~finished[t])

        for q in range(self.model.num_value_nets):
          value_loss += self.rho**t * soft_crossentropy(
              q_logits[q, t], td_targets[t],
              self.model.symlog_min,
              self.model.symlog_max,
              self.model.num_bins).mean(where=~finished[t])

      if self.model.predict_continues:
        continue_loss = optax.sigmoid_binary_cross_entropy(
            continue_logits, 1 - terminated).mean()
      else:
        continue_loss = 0

      consistency_loss = consistency_loss / self.horizon
      reward_loss = reward_loss / self.horizon
      value_loss = value_loss / self.horizon / self.model.num_value_nets
      total_loss = (
          self.consistency_coef * consistency_loss +
          self.reward_coef * reward_loss +
          self.value_coef * value_loss +
          self.continue_coef * continue_loss
      )

      return total_loss, {
          'consistency_loss': consistency_loss,
          'reward_loss': reward_loss,
          'value_loss': value_loss,
          'continue_loss': continue_loss,
          'total_loss': total_loss,
          'zs': zs
      }

    # Update world model
    (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads), model_info = jax.grad(
        world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None)
    zs = model_info.pop('zs')

    new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
    new_dynamics_model = self.model.dynamics_model.apply_gradients(
        grads=dynamics_grads)
    new_reward_model = self.model.reward_model.apply_gradients(
        grads=reward_grads)
    new_value_model = self.model.value_model.apply_gradients(
        grads=value_grads)
    new_target_value_model = self.model.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.model.target_value_model.params,
            self.tau))
    if self.model.predict_continues:
      new_continue_model = self.model.continue_model.apply_gradients(
          grads=continue_grads)
    else:
      new_continue_model = self.model.continue_model

    # Update policy
    def policy_loss_fn(params: Dict):
      action_key, Q_key, ensemble_key = jax.random.split(policy_key, 3)
      actions, _, _, log_probs = self.model.sample_actions(                    # log_probs (horizon+1, batch_size)
          zs, params, key=action_key)

      # Compute Q-values
      Qs, _ = self.model.Q(zs, actions, new_value_model.params, key=Q_key)     # Qs (num_qs, horizon+1, batch_size)

      # Yuan: added to conform to the original implementation
      # Sample two Q-values from the target ensemble
      inds = jax.random.choice(ensemble_key,
                             jnp.arange(0, self.model.num_value_nets),
                             shape=(2, ), replace=False)

      Q = Qs[inds].mean(axis=0)                                                # Q (horizon+1, batch_size)
      # Update and apply scale
      scale = percentile_normalization(Q[0], self.scale).clip(1, None)
      Q = Q / sg(scale)                                                        # Q (horizon+1, batch_size)

      # Compute policy objective (equation 4)
      rho = self.rho ** jnp.arange(self.horizon+1)
      policy_loss_entropy = self.entropy_coef * log_probs
      policy_loss = ((self.entropy_coef * log_probs -
                     Q).mean(axis=1) * rho).mean()
      return policy_loss, {'policy_loss': policy_loss, 'policy_scale': scale, }

    # def policy_loss_fn(params: Dict): # original
    #   action_key, Q_key = jax.random.split(policy_key, 2)
    #   actions, _, _, log_probs = self.model.sample_actions(
    #       zs, params, key=action_key)

    #   # Compute Q-values
    #   Qs, _ = self.model.Q(zs, actions, new_value_model.params, key=Q_key)
    #   Q = Qs.mean(axis=0)
    #   # Update and apply scale
    #   scale = percentile_normalization(Q[0], self.scale).clip(1, None)
    #   Q = Q / sg(scale)

    #   # Compute policy objective (equation 4)
    #   rho = self.rho ** jnp.arange(self.horizon+1)
    #   policy_loss = ((self.entropy_coef * log_probs -
    #                  Q).mean(axis=1) * rho).mean()
    #   return policy_loss, {'policy_loss': policy_loss, 'policy_scale': scale}

    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.model.policy_model.params)
    new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

    # Update model
    new_agent = self.replace(model=self.model.replace(
        encoder=new_encoder,
        dynamics_model=new_dynamics_model,
        reward_model=new_reward_model,
        value_model=new_value_model,
        policy_model=new_policy,
        target_value_model=new_target_value_model,
        continue_model=new_continue_model),
        scale=policy_info['policy_scale'])
    info = {**model_info, **policy_info}

    return new_agent, info

  @jax.jit
  def estimate_value(self, z: jax.Array, actions: jax.Array, key: PRNGKeyArray) -> jax.Array:
    G, discount = 0.0, 1.0
    for t in range(self.horizon):
      reward, _ = self.model.reward(
          z, actions[t], self.model.reward_model.params)
      z = self.model.next(z, actions[t], self.model.dynamics_model.params)
      G += discount * reward.astype(jnp.float32)

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(self.model.continue_model.apply_fn(
            {'params': self.model.continue_model.params}, z)).squeeze(-1) > 0.5
      else:
        continues = 1.0

      discount *= self.discount * continues

    action_key, Q_key, ensemble_key = jax.random.split(key, 3)
    next_action = self.model.sample_actions(
        z, self.model.policy_model.params, key=action_key)[0]

    Qs, _ = self.model.Q(
        z, next_action, self.model.value_model.params, key=Q_key)
    
    # Yuan: conform to the original tdmpc2
    # Sample two Q-values from the target ensemble
    inds = jax.random.choice(ensemble_key,
                             jnp.arange(0, self.model.num_value_nets),
                             shape=(2, ), replace=False)
    Q = Qs[inds].mean(axis=0)
    return sg(G + discount * Q)

#   @jax.jit
#   def estimate_value(self, z: jax.Array, actions: jax.Array, key: PRNGKeyArray) -> jax.Array: # original
#     G, discount = 0.0, 1.0
#     for t in range(self.horizon):
#       reward, _ = self.model.reward(
#           z, actions[t], self.model.reward_model.params)
#       z = self.model.next(z, actions[t], self.model.dynamics_model.params)
#       G += discount * reward.astype(jnp.float32)

#       if self.model.predict_continues:
#         continues = jax.nn.sigmoid(self.model.continue_model.apply_fn(
#             {'params': self.model.continue_model.params}, z)).squeeze(-1) > 0.5
#       else:
#         continues = 1.0

#       discount *= self.discount * continues

#     action_key, Q_key = jax.random.split(key, 2)
#     next_action = self.model.sample_actions(
#         z, self.model.policy_model.params, key=action_key)[0]

#     Qs, _ = self.model.Q(
#         z, next_action, self.model.value_model.params, key=Q_key)
#     Q = Qs.mean(axis=0)
#     return sg(G + discount * Q)

  @jax.jit
  def td_target(self, next_z: jax.Array, reward: jax.Array, terminal: jax.Array,
                key: PRNGKeyArray) -> jax.Array:
    """
    next_z (horizon, batch_sz, latent_dim)
    reward (horizon, batch_sz)
    Return: Q (horizon, batch_sz)
    """
    action_key, ensemble_key, Q_key = jax.random.split(key, 3)
    next_action = self.model.sample_actions(
        next_z, self.model.policy_model.params, key=action_key)[0]

    # Sample two Q-values from the target ensemble
    inds = jax.random.choice(ensemble_key,
                             jnp.arange(0, self.model.num_value_nets),
                             shape=(2, ), replace=False)
    Qs, _ = self.model.Q(
        next_z, next_action, self.model.target_value_model.params, key=Q_key)
    Q = Qs[inds].min(axis=0)
    return sg(reward + (1 - terminal) * self.discount * Q)
