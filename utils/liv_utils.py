import os
import copy
from os.path import expanduser
import omegaconf
import gdown
import hydra
import torch
from huggingface_hub import hf_hub_download
from torch.hub import load_state_dict_from_url


device = "cuda" if torch.cuda.is_available() else "cpu"


VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size",
              "l2weight", "l1weight", "num_negatives"]


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "models.LIV"
    config["device"] = device
    return config.agent


def load_liv(model_id="resnet50"):
    """
    model_config = {
        'save_snapshot': True,
        'load_snap': '',
        'dataset': 'ego4d',
        'num_workers': 10,
        'batch_size': 32,
        'train_steps': 2000000,
        'eval_freq': 20000,
        'seed': 1,
        'device': 'cuda',
        'lr': 0.0001,
        'wandbproject': None,
        'wandbuser': None,
        'doaug': 'rctraj',
        'agent': {
            '_target_': 'models.model_vip.VIP',
            'device': '${device}',
            'lr': '${lr}',
            'hidden_dim': 1024,
            'size': 50,
            'l2weight': 0.001,
            'l1weight': 0.001,
            'gamma': 0.98,
            'bs': '${batch_size}'
        }
    }

    clean_config = {
        '_target_': 'vip.VIP',
        'device': '${device}',
        'lr': '${lr}',
        'hidden_dim': 1024,
        'size': 50,
        'l2weight': 0.001,
        'l1weight': 0.001
    }
    """
    base_dir = os.path.join(expanduser("~"), ".liv")
    os.makedirs(os.path.join(base_dir, model_id), exist_ok=True)

    folder_dir = os.path.join(base_dir, model_id)
    model_dir = os.path.join(base_dir, model_id, "model.pt")
    config_dir = os.path.join(base_dir, model_id, "config.yaml")

    try: 
        hf_hub_download(repo_id="jasonyma/LIV",
                        filename="model.pt",
                        local_dir=folder_dir)
        hf_hub_download(repo_id="jasonyma/LIV",
                        filename="config.yaml",
                        local_dir=folder_dir)
    except:
        model_url = "https://drive.google.com/uc?id=1l1ufzVLxpE5BK7JY6ZnVBljVzmK5c4P3"
        config_url = "https://drive.google.com/uc?id=1GWA5oSJDuHGB2WEdyZZmkro83FNmtaWl"
        if not os.path.exists(model_dir):
            gdown.download(model_url, model_dir, quiet=False)
            gdown.download(config_url, config_dir, quiet=False)
        else:
            load_state_dict_from_url(model_url,
                                     folder_dir,
                                     map_location=torch.device(device))
            load_state_dict_from_url(config_url, folder_dir)

    model_config = omegaconf.OmegaConf.load(config_dir)
    clean_config = cleanup_config(model_config)

    liv_model = hydra.utils.instantiate(clean_config)
    liv_model = torch.nn.DataParallel(liv_model)
    liv_state_dict = torch.load(model_dir, map_location=torch.device(device))["liv"]
    liv_model.load_state_dict(liv_state_dict)
    liv_model.eval()
    return liv_model
