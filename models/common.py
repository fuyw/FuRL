from typing import Any, Callable, Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn


###################
# Utils Functions #
###################
class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


################
# Actor Critic #
################
class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    min_scale: float = 1e-3

    def setup(self):
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim)
        self.std_layer = nn.Dense(self.act_dim)

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(
            seed=rng)

        return mean_action * self.max_action, sampled_action * self.max_action, logp

    def get_logprob(self, observation, action):
        x = self.net(observation)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        log_prob = action_distribution.log_prob(raw_action).sum(-1)
        log_prob -= 2 * (jnp.log(2) - raw_action -
                         jax.nn.softplus(-2 * raw_action)).sum(-1)
        return log_prob, mu, std


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1

    def setup(self):
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.out_layer = nn.Dense(self.output_dim)

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze()


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        VmapCritic = nn.vmap(Critic,
                             variable_axes={"params": 0},
                             split_rngs={"params": True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims, self.output_dim)(observations,
                                                           actions)
        return qs


####################
# Vectorized Agent #
####################
class EnsembleDense(nn.Module):
    ensemble_num: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            "kernel", self.kernel_init,
            (self.ensemble_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init,
                              (self.ensemble_num, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class EnsembleCritic(nn.Module):
    ensemble_num: int
    hid_dim: int = 256

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc2")
        self.l3 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=1,
                                name="fc3")

    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        x = self.l3(x)
        return x.squeeze(-1)


class EnsembleDoubleCritic(nn.Module):
    ensemble_num: int
    hid_dim: int = 256

    def setup(self):
        self.q1 = EnsembleCritic(self.ensemble_num, self.hid_dim)
        self.q2 = EnsembleCritic(self.ensemble_num, self.hid_dim)

    def __call__(self, observations, actions):
        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        return q1, q2


class EnsembleActor(nn.Module):
    ensemble_num: int
    act_dim: int
    hid_dim: int = 256
    max_action: float = 1.0
    min_scale: float = 1e-3

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc2")
        self.mu_layer = EnsembleDense(ensemble_num=self.ensemble_num,
                                      features=self.act_dim,
                                      name="mu")
        self.std_layer = EnsembleDense(ensemble_num=self.ensemble_num,
                                       features=self.act_dim,
                                       name="std")

    def __call__(self, observation: jnp.ndarray):
        x = nn.relu(self.l1(observation))
        x = nn.relu(self.l2(x))
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        return mean_action, action_distribution

    def get_logprob(self, observation, action):
        x = nn.relu(self.l1(observation))
        x = nn.relu(self.l2(x))
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        log_prob = action_distribution.log_prob(raw_action).sum(-1)
        log_prob -= 2 * (jnp.log(2) - raw_action -
                         jax.nn.softplus(-2 * raw_action)).sum(-1)
        return log_prob


class EnsembleScalar(nn.Module):
    init_value: jnp.ndarray

    def setup(self):
        self.value = self.param("value", lambda x: jnp.array(self.init_value))

    def __call__(self):
        return self.value


class EnsembleMLP(nn.Module):
    ensemble_num: int
    hid_dim: int = 256

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc2")

    def __call__(self, x):
        x = nn.relu(self.l1(x))
        x = self.l2(x)
        return x
