
# HGAIL
- This repo contains implementations of the following, built on top of [rllab](https://github.com/openai/rllab):
    + [generative adversarial imitation learning](https://arxiv.org/abs/1606.03476) (GAIL) 
    + [InfoGAIL](https://arxiv.org/abs/1703.08840)
    + Hierarchical GAIL
- The main goal of these implementations is for them to be easy to use with rllab environments
- The repo is called `hgail` because the original purpose was implementing the hierarchical GAIL variant, though it has since then primarily been used for the GAIL implementation

# Install
- install [rllab](https://github.com/openai/rllab)
- clone this repo, and run setup.py in develop mode
```bash
git clone https://github.com/sisl/hgail.git
cd hgail
python setup.py develop
```
- after installing, run the tests in the `tests` directory to check that everything is correctly installed
 ```bash
 cd tests
 python runtests.py
 ```

# Examples
- for an artificial example on cartpole, see [`scripts/README.md`](scripts/README.md) 
- for an actual application, see this [repo](https://github.com/sisl/ngsim_env) on learning human driver models with gail