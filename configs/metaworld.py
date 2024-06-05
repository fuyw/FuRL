import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.env_name = "peg-insert-side-v2-goal-observable"
    config.camera_id = 2
    config.residual = False
    config.eval_episodes = 100
    config.start_timesteps = 10000
    config.max_timesteps = int(1e6)
    config.decay_timesteps = int(7.5e5)
    config.eval_freq = config.max_timesteps // 10
    config.log_freq = config.max_timesteps // 100
    config.ckpt_freq = config.max_timesteps // 10
    config.lr = 1e-4
    config.seed = 0
    config.tau = 0.01
    config.gamma = 0.99
    config.batch_size = 256
    config.hidden_dims = (256, 256)
    config.initializer = "orthogonal"
    config.exp_name = "furl"

    # relay
    config.relay_threshold = 2500
    config.expl_noise = 0.2

    # fine-tune
    config.rho = 0.05
    config.gap = 10
    config.crop = False
    config.l2_margin = 0.25
    config.cosine_margin = 0.25
    config.embed_buffer_size = 20000

    return config
