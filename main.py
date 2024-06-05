from absl import app, flags
from ml_collections import config_flags
import os
import experiments

config_flags.DEFINE_config_file("config", default="configs/metaworld.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config

    try:
        if config.exp_name == "oracle":
            experiments.run_oracle.evaluate(config)

        elif config.exp_name == "sac":
            experiments.train_sac.train_and_evaluate(config)

        elif config.exp_name == "liv":
            experiments.train_liv.train_and_evaluate(config)

        elif config.exp_name == "relay":
            experiments.train_relay.train_and_evaluate(config)

        elif config.exp_name == "furl":
            experiments.train_furl.train_and_evaluate(config)

    except KeyboardInterrupt as e:
        print("Skip to the next experiment.")


if __name__ == '__main__':
    app.run(main)
