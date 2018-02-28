
# Walkthrough
1. train the expert policy
```
python train.py
```
- this will train an expert agent in a cartpole environment

2. collect trajectories from the expert 
```
python simulate.py --mode collect --itr 95 --n_traj 500
```
- this will collect trajectories and store them in data/trajectories
- by default the last itr in training is itr 95
- this collects n_traj episodes worth of trajectories (500 here)

3. run imitation learning 
```
python imitate.py
```
- this will load the trajectories, and train a GAIL model

4. visualize the GAIL model
```
python simulate.py --visualize
```
- this will visualize the GAIL model acting in the environment

5. evaluate the GAIL model 
```
python simulate.py --evaluate --n_traj 100 --itr 195
```
- simulate the GAIL model for n_traj episodes
    + this currently just prints the average reward
- `--itr 195` tells the script to use the policy saved after iteration 195