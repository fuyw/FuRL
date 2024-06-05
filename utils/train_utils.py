import logging
import git
import jax
import numpy as np
from flax.core import FrozenDict


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def log_git(config):
    config.unlock()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config["commit"] = sha


def target_update(params: FrozenDict,
                  target_params: FrozenDict,
                  tau: float) -> FrozenDict:

    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau * param + (1 - tau) * target_param

    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params
