# FuRL

## Environment Setup

Install the conda env via:

```shell
conda create --name furl python==3.11
conda activate furl
pip install -r requirements.txt
```

## Training

### Generating Expert Dataset

An optional setting in FuRL is to use a goal image to accelerate the exploration before we collected the first successful trajectory.

```script
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=oracle
```

The oracle trajectory data will be saved in `data/oracle`.

### Example on Fixed-goal Task

```
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=furl
```

### Example on Random-goal Task

```
python main.py --config.env_name=door-open-v2-goal-observable --config.exp_name=furl
```

## Paper

[**FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning**](https://arxiv.org/pdf/2406.00645)

Yuwei Fu, Haichao Zhang, Di Wu, Wei Xu, Benoit Boulet

*International Conference on Machine Learning* (ICML), 2024

## Cite

Please cite our work if you find it useful:

```txt
@InProceedings{fu2024,
  title = {FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning},
  author = {Yuwei Fu and Haichao Zhang and Di Wu and Wei Xu and Benoit Boulet},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024}
}
```
