from typing import Any, Dict, Sequence
import os
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
from utils import Batch, target_update
from models import Actor, DoubleCritic, Scalar, MLP


class FuRLAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 rho: float = 0.1,
                 margin: float = 0.1,  
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 ckpt_dir: str = None,
                 text_embedding: jnp.ndarray = None,
                 goal_embedding: jnp.ndarray = None,
                 hidden_dims: Sequence[int] = (256, 256)):
        self.lr = lr
        self.tau = tau
        self.rho = rho
        self.gamma = gamma
        self.margin = margin
        self.max_action = max_action
        self.target_entropy = -act_dim / 2
        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, proj_key = jax.random.split(self.rng, 4)
        self.goal_embedding = goal_embedding
        self.text_embedding = text_embedding

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Create the optimizer
        actor_tx = optax.adam(lr)
        critic_tx = optax.adam(lr)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           hidden_dims=hidden_dims)
        actor_params = self.actor.init(actor_key,
                                       actor_key,
                                       dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=actor_tx)

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims)
        critic_params = self.critic.init(critic_key,
                                         dummy_obs,
                                         dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=critic_tx)

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr))

        # Checkpoint
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.checkpointer = ocp.StandardCheckpointer()

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray) -> jnp.ndarray:
        mean_action, sampled_action, _ = self.actor.apply({"params": params},
                                                          rng,
                                                          observation)
        return mean_action, sampled_action

    def sample_action(self,
                      observation: np.ndarray,
                      eval_mode: bool = False) -> np.ndarray:
        self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action = self._sample_action(
            self.actor_state.params, sample_rng, observation)
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        return action.clip(-self.max_action, self.max_action)

    def actor_alpha_train_step(self, batch: Batch, key: Any,
                               alpha_state: train_state.TrainState,
                               actor_state: train_state.TrainState,
                               critic_state: train_state.TrainState):

        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict, actor_params: FrozenDict,
                    rng: Any, observation: jnp.ndarray):
            # sample action with actor
            _, sampled_action, logp = self.actor.apply(
                {"params": actor_params}, rng, observation)

            # compute alpha loss
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha = jnp.exp(log_alpha)
            alpha_loss = -alpha * jax.lax.stop_gradient(
                logp + self.target_entropy)

            # stop alpha gradient
            alpha = jax.lax.stop_gradient(alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # return info
            actor_alpha_loss = actor_loss + alpha_loss
            log_info = {
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": alpha,
                "logp": logp
            }
            return actor_alpha_loss, log_info

        # compute gradient with vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1),
                                              has_aux=True),
                           in_axes=(None, None, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(alpha_state.params, actor_state.params,
                                       keys, batch.observations)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        # Update TrainState
        alpha_grads, actor_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        return new_alpha_state, new_actor_state, log_info

    def critic_train_step(self,
                          batch: Batch,
                          vlm_rewards: jnp.ndarray,
                          key: Any,
                          alpha: float,
                          actor_state: train_state.TrainState,
                          critic_state: train_state.TrainState,
                          critic_target_params: FrozenDict):

        frozen_actor_params = actor_state.params

        def loss_fn(params: FrozenDict, rng: Any, observation: jnp.ndarray,
                    action: jnp.ndarray, reward: jnp.ndarray,
                    next_observation: jnp.ndarray, discount: jnp.ndarray):

            # current q value
            q1, q2 = self.critic.apply({"params": params}, observation, action)

            # next q value
            _, next_action, logp_next_action = self.actor.apply(
                {"params": frozen_actor_params}, rng, next_observation)
            next_q1, next_q2 = self.critic.apply(
                {"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2) - alpha * logp_next_action

            # target q value
            target_q = reward + self.gamma * discount * next_q

            # td error
            critic_loss1 = (q1 - target_q)**2
            critic_loss2 = (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2
            log_info = {
                "critic_loss": critic_loss,
                "q": q1,
            }
            return critic_loss, log_info

        # compute gradient with vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True),
                           in_axes=(None, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        # reward shaping
        rewards = batch.rewards + self.rho * vlm_rewards
        (_, log_info), grads = grad_fn(critic_state.params,
                                       keys,
                                       batch.observations,
                                       batch.actions,
                                       rewards, 
                                       batch.next_observations,
                                       batch.discounts)
        extra_log_info = {"q_max": log_info["q"].max(), "rvlm_reward": rewards.mean()}
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)

        # Update TrainState
        new_critic_state = critic_state.apply_gradients(grads=grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        return new_critic_state, new_critic_target_params, log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   vlm_rewards: jnp.ndarray,
                   key: Any,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        key1, key2 = jax.random.split(key)
        new_alpha_state, new_actor_state, actor_log_info = self.actor_alpha_train_step(
            batch, key1, alpha_state, actor_state, critic_state)
        alpha = actor_log_info["alpha"]
        new_critic_state, new_critic_target_params, critic_log_info = self.critic_train_step(
            batch, vlm_rewards, key2, alpha, actor_state, critic_state,
            critic_target_params)
        log_info = {**actor_log_info, **critic_log_info}
        return new_alpha_state, new_actor_state, new_critic_state, \
            new_critic_target_params, log_info

    def update(self, batch: Batch, vlm_rewards: jnp.ndarray):
        self.rng, key = jax.random.split(self.rng, 2)
        (self.alpha_state,
         self.actor_state,
         self.critic_state,
         self.critic_target_params,
         log_info) = self.train_step(batch,
                                     vlm_rewards,
                                     key,
                                     self.alpha_state,
                                     self.actor_state,
                                     self.critic_state,
                                     self.critic_target_params)
        return log_info

    def save(self, cnt: int = 0):
        params = {"actor": self.actor_state.params,
                  "critic": self.critic_state.params}
        self.checkpointer.save(f"{self.ckpt_dir}/{cnt}",
                               params,
                               force=True)

    def load(self, ckpt_dir: str, cnt: int = 0):
        raw_restored = self.checkpointer.restore(f"{ckpt_dir}/{cnt}")
        actor_params = raw_restored["actor"]
        critic_params = raw_restored["critic"]

        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(self.lr)) 
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))
        self.critic_target_params = critic_params
