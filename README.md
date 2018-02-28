
# GAIL
This repo contains an implementations of the following, built on top of [rllab](https://github.com/openai/rllab)
- [generative adversarial imitation learning](https://arxiv.org/abs/1606.03476) (GAIL) 
- [InfoGAIL](https://arxiv.org/abs/1703.08840)

# Install
- install rllab
```bash
# note: the following asssumes you have conda (miniconda at least) installed on your system
# clone rllab repo
cd rllab/scripts
sh ./setup_osx.sh # change to the linux script if you run linux
```

- run
```bash
python setup.py develop
```

- TBD, but for now add the package to your .bash_profile manually if you'd like to use:
```bash
export PYTHONPATH=:/path/to/hgail$PYTHONPATH
```

## requirements
- TBD

# Scripts & Examples

## Critic
- See `scripts/examples/critic_toy_data.py`

## Imitation
- Initial scripts `train.py`, `simulate.py`, `imitate.py`
1. run `train.py`
    + this will train an initial agent in some hard coded environment
2. run `simulate.py` to collect expert trajectories from the trained agent
    + this will store the trajectories in an experiment directory
3. run `imitate.py` to train a GAIL or InfoGAIL agent
    + this will train a gail model
4. run `simulate.py`, this time changing the policy to the gail one
    + this will simulate episodes in the saved env with the policy
    + you can visualize this or collect average stats

# Package Outline

## GAIL
- Main class. Inherits from TRPO and serves to interject critic rewards into the trajectories of the agent. Also orchestrates training of critic and recognition networks. See gail.py

## critic 
- [Wasserstein critic](https://arxiv.org/abs/1704.00028). See critic.py
