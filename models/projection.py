import jax
import jax.numpy as jnp
import functools
import optax
import orbax.checkpoint as ocp

from flax import linen as nn
from flax.training import train_state
from models import MLP


class Projection(nn.Module):
    def setup(self):
        self.text_encoder = MLP(hidden_dims=(256, 64), activate_final=False)
        self.image_encoder = MLP(hidden_dims=(256, 64), activate_final=False)

    def __call__(self, text_embedding, image_embedding):
        proj_text_embedding = self.text_encoder(text_embedding)
        proj_image_embedding = self.image_encoder(image_embedding)
        return proj_text_embedding, proj_image_embedding

    def encode_image(self, image_embeddings):
        return self.image_encoder(image_embeddings)

    def encode_text(self, text_embedding):
        return self.text_encoder(text_embedding)


class RewardModel:
    def __init__(self,
                 seed: int = 42,
                 lr: float = 1e-4,
                 margin: float = 0.1,
                 emb_dim: int = 1024,
                 ckpt_dir: str = None,
                 text_embedding: jnp.ndarray = None,
                 goal_embedding: jnp.ndarray = None):
        self.lr = lr
        self.margin = margin
        self.text_embedding = text_embedding
        self.goal_embedding = goal_embedding
        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng, 2)
        dummy_emb = jnp.ones([1, emb_dim], dtype=jnp.float32)

        self.proj = Projection()
        proj_params = self.proj.init(key,
                                     jnp.ones([1, 1024], dtype=jnp.float32),
                                     dummy_emb)["params"]
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(lr))

        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.checkpointer = ocp.StandardCheckpointer()

    @functools.partial(jax.jit, static_argnames=("self"))
    def get_vlm_reward(self, proj_state, img_embeddings):
        proj_img_embeddings = self.proj.apply(
            {"params": proj_state.params}, img_embeddings,
            method=self.proj.encode_image)
        proj_text_embedding = self.proj.apply(
            {"params": proj_state.params}, self.text_embedding,
            method=self.proj.encode_text)
        cosine_similarity = optax.cosine_similarity(proj_img_embeddings,
                                                    proj_text_embedding)
        return cosine_similarity

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_pos_step(self,
                       pos_embeddings,
                       neg_embeddings,
                       lag_embeddings,
                       proj_state):
        def loss_fn(params):
            proj_text_embedding = self.proj.apply(
                {"params": params}, self.text_embedding,
                method=self.proj.encode_text)

            proj_pos_embeddings = self.proj.apply(
                {"params": params}, pos_embeddings,
                method=self.proj.encode_image)
            proj_neg_embeddings = self.proj.apply(
                {"params": params}, neg_embeddings,
                method=self.proj.encode_image)
            proj_lag_embeddings = self.proj.apply(
                {"params": params}, lag_embeddings,
                method=self.proj.encode_image)

            pos_cosine = optax.cosine_similarity(proj_text_embedding,
                                                 proj_pos_embeddings)
            neg_cosine = optax.cosine_similarity(proj_text_embedding, 
                                                 proj_neg_embeddings)
            lag_cosine = optax.cosine_similarity(proj_text_embedding, 
                                                 proj_lag_embeddings)

            # pos-neg: pos_cosine > lag_cosine > negative_cosine
            neg_mask = (neg_cosine - pos_cosine + self.margin) > 0
            neg_loss = neg_mask * (neg_cosine - pos_cosine)

            # pos-pos: pos_cosine > lag_cosine
            pos_mask = (lag_cosine - pos_cosine + self.margin) > 0
            pos_loss = pos_mask * (lag_cosine - pos_cosine)
            total_loss = pos_loss.mean() + neg_loss.mean()
            log_info = {
                "pos_cosine": pos_cosine.mean(),
                "pos_cosine_max": pos_cosine.max(),
                "pos_cosine_min": pos_cosine.min(),

                "neg_cosine": neg_cosine.mean(),
                "neg_cosine_max": neg_cosine.max(),
                "neg_cosine_min": neg_cosine.min(),

                "lag_cosine": lag_cosine.mean(),
                "lag_cosine_max": lag_cosine.max(),
                "lag_cosine_min": lag_cosine.min(),

                "neg_num": neg_mask.sum(),
                "neg_loss": neg_loss.mean(),
                "neg_loss_max": neg_loss.max(),

                "pos_num": pos_mask.sum(),
                "pos_loss": pos_loss.mean(),
                "pos_loss_max": pos_loss.max(),
            }
            return total_loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)        
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_neg_step(self,
                       batch,
                       proj_state):
        def loss_fn(params):
            proj_text_embedding = self.proj.apply(
                {"params": params}, self.text_embedding,
                method=self.proj.encode_text)

            proj_embeddings = self.proj.apply(
                {"params": params}, batch.embeddings,
                method=self.proj.encode_image)

            # cosine similarity
            cosine = optax.cosine_similarity(proj_text_embedding, proj_embeddings)
            cosine_delta = cosine.reshape(-1, 1) - cosine.reshape(1, -1)
  
            loss = (nn.relu(-cosine_delta + self.margin) * batch.masks).sum(-1).mean()
            log_info = {"pos_loss": loss, "vlm_rewards": cosine}
            return loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    def update_neg(self, batch):
        self.proj_state, log_info = self.train_neg_step(batch, self.proj_state) 
        return log_info

    def update_pos(self, batch):  
        self.proj_state, log_info = self.train_pos_step(batch.pos_embeddings,
                                                        batch.neg_embeddings,
                                                        batch.lag_embeddings,
                                                        self.proj_state) 
        return log_info

    def save(self, cnt):
        self.checkpointer.save(f"{self.ckpt_dir}/{cnt}",
                               {"proj": self.proj_state.params},
                               force=True)

    def load(self, ckpt_dir: str, cnt: int = 0):
        raw_restored = self.checkpointer.restore(f"{ckpt_dir}/{cnt}")
        proj_params = raw_restored["proj"]
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(self.lr))
