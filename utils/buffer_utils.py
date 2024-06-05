import collections
import jax
import logging
import numpy as np
from flax.core import FrozenDict


# basic batch
Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


FinetuneBatch = collections.namedtuple(
    "FinetuneBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "embeddings"])


MaskBatch = collections.namedtuple(
    "MaskBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "embeddings", "masks"])


VLMBatch = collections.namedtuple(
    "VLMBatch",
    ["observations", "actions", "rewards", "vlm_rewards", "discounts", "next_observations"])


EmbeddingBatch = collections.namedtuple(
    "EmbeddingBatch",
    ["pos_embeddings", "neg_embeddings", "lag_embeddings"])


class ReplayBuffer:

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def save(self, fname: str):
        np.savez(fname,
                 observations=self.observations,
                 actions=self.actions,
                 next_observations=self.next_observations,
                 rewards=self.rewards,
                 discounts=self.discounts)


class VLMBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.vlm_rewards = np.zeros(max_size)
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            vlm_reward: float,
            reward: float,
            done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.vlm_rewards[self.ptr] = vlm_reward
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        rewards = self.rewards[idx]
        vlm_rewards = self.vlm_rewards[idx]
        batch = VLMBatch(observations=self.observations[idx],
                       actions=self.actions[idx],
                       rewards=rewards,
                       vlm_rewards=vlm_rewards,
                       discounts=self.discounts[idx],
                       next_observations=self.next_observations[idx])
        return batch


class DistanceBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 emb_dim: int = 1024,
                 max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.rewards = np.zeros(max_size)
        self.next_observations = np.zeros((max_size, obs_dim))
        self.discounts = np.zeros(max_size)
        self.embeddings = np.zeros((max_size, emb_dim))
        self.distances = np.zeros((max_size))

    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray, 
            reward: float,
            done: float,
            embedding: np.ndarray,
            distance: float = 0):

        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.embeddings[self.ptr] = embedding
        self.distances[self.ptr] = distance

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_with_mask(self, batch_size: int, l2_margin: float = 0.05) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        distance = self.distances[idx]

        l2_delta = distance.reshape(-1, 1) - distance.reshape(1, -1)
        masks = (l2_delta < -l2_margin).astype(np.float32)

        batch = MaskBatch(observations=self.observations[idx],
                          actions=self.actions[idx],
                          rewards=self.rewards[idx],
                          discounts=self.discounts[idx],
                          next_observations=self.next_observations[idx],
                          embeddings=self.embeddings[idx],
                          masks=masks)

        return batch

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = FinetuneBatch(observations=self.observations[idx],
                              actions=self.actions[idx],
                              rewards=self.rewards[idx],
                              discounts=self.discounts[idx],
                              next_observations=self.next_observations[idx],
                              embeddings=self.embeddings[idx])

        return batch


class EmbeddingBuffer:
    def __init__(self,
                 emb_dim: int,
                 gap: int = 10,
                 max_size: int = int(1e5)):
        self.gap = gap
        self.max_size = max_size

        self.pos_ptr = 0
        self.pos_size = 0
        self.pos_embeddings = np.zeros((max_size, emb_dim))

        self.neg_ptr = 0
        self.neg_size = 0
        self.neg_embeddings = np.zeros((max_size, emb_dim))

        self.valid_ptr = 0
        self.valid_size = 0
        self.valid_idxes = np.zeros(max_size, dtype=np.int32)

    def add(self,
            embedding: np.ndarray,
            success: bool = False,
            valid: bool = False):
        if success:
            self.pos_embeddings[self.pos_ptr] = embedding
            if valid:
                self.valid_idxes[self.valid_ptr] = self.pos_ptr
                self.valid_ptr = (self.valid_ptr + 1) % self.max_size
                self.valid_size = min(self.valid_size + 1, self.max_size)
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size
            self.pos_size = min(self.pos_size + 1, self.max_size)
        else:
            self.neg_embeddings[self.neg_ptr] = embedding
            self.neg_ptr = (self.neg_ptr + 1) % self.max_size
            self.neg_size = min(self.neg_size + 1, self.max_size)

    def sample(self, batch_size):
        neg_idx = np.random.randint(0, self.neg_size, size=batch_size)
        valid_idx = np.random.randint(0, self.valid_size, size=batch_size)
        pos_idx = self.valid_idxes[valid_idx]
        lag_idx = (pos_idx - self.gap) % self.valid_size

        pos_embeddings = self.pos_embeddings[pos_idx]
        lag_embeddings = self.pos_embeddings[lag_idx]
        neg_embeddings = self.neg_embeddings[neg_idx]
        return EmbeddingBatch(pos_embeddings=pos_embeddings,
                              lag_embeddings=lag_embeddings,
                              neg_embeddings=neg_embeddings)

    def save(self, fdir):
        np.savez(fdir,
                 pos_embeddings=self.pos_embeddings,
                 neg_embeddings=self.neg_embeddings,
                 pos_ptr=self.pos_ptr,
                 pos_size=self.pos_size,
                 neg_ptr=self.neg_ptr,
                 neg_size=self.neg_size,
                 valid_ptr=self.valid_ptr,
                 valid_size=self.valid_size,
                 valid_idxes=self.valid_idxes)
