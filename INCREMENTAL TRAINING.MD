# Self Play Incremental Training

The following training method is a part of my research MEng researh project. Though agents have camera, the training was made using state observations.

## Abstract

We first launched the agent in the full 3D environment. But the environment was too challenging for it to solve to be able to train.
We then tried to train the training by projecting the 3D space in  2D (The agent was still be reading x,y,z axes but the z values remains 0 for all mobile objects in the environment). We notice then that the agent was training the same way as it does in the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment.
We shifted the trained agent in the 2D fashion in the full 3D environment (from z = 0 to  z = 24), the agent failed mastering the 3D environment.
We shifted the trained agent in 2D to a tiny 3D envirnment (from z = 0 to z = 4) and notice that after losing performance a bit the agent was able to adapt and training in the reduced 3D dimension. 
We decided to shift again by adding 4 units to the z axis (from z= 4 to z = 8) the agent was still adapting. We continued the process untill the full depth (z = 24) and the agent was finally able to play in the full game space and able to last the maximim number of timesteps of an episode. Hence the incremental fashion that made a PPO agent to solve a 3D versio of the slime volley game.

## Scripts

n the training [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) there are 3 main attributes that make the incremental training a success:

```python
UPGR_LEN  # The performance threshold in terms of mean episode legnth
```
Waiting for the agent to reach the optimal policy before adding more difficulty is time consuming. We set a threshold performance value such that if the agent reaches that threshold we can move it in a upper level of difficulty. We used the average time it lasts in the game as the performance indicator. One would prefer to use the reward as a performance indicator.

```python
n_update # parameter fed to the function setup() of the WORLD class
```
Defines the number of time the environment z axis will be incremented during the training if initially z = 0. It helps to compute the *incremetation step* = 24/ n_update. For example if n_update = 6, step  = 24/6 = 4. This mean every time the threshold performance is reached the z axis will be increased by 4 units.


### The environment setup

We are just expalining the main attribute of the environment that help in the incremental setting.

```python
env = VolleyBotSelfPlayEnv()

# training = True means during training the ball will always be launched on the learning, this speeds up the training time
env.training = True 

# update = True, means the z axis will be incremented each time the performance threshold is reached during evaluation
env.update = True # Defautly to False, if True the z axis will be incremented each time the performance threshold is

# stuck = True, means no incrementation learning, the initial depth will stay fixed during the whole training
env.world.stuck = False 

# n_update precises the value of the step at each incrementation, step = 24/n_update
# init_depth precises the value of the z axis at the initialization, it's not oblige to start with 0
env.world.setup( n_update = 6, init_depth = 4) 
                                              
env.seed(SEED)
```

### Training Script

We used the stablebaselines [PPO2](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py) to train but the environment is independent of the training algorithm and framework. Stablebaselines3 can work as well with the script.

In the training [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) one of the main parameters to worry about is:
```python
UPGR_LEN # The performance threshold in terms of mean episode legnth
```
Once the average episode legnth during evaluation exceed *UPGR_LEN* the environment get wider according the following the predefine step.


The success of the incremental of the training relay on those 3 parameters: *init_depth, upgr_len and n_update*.

