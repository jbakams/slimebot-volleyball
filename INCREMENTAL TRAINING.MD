# Self Play Incremental Training

The following training method is a part of my research MEng researh project. Though agents have camera, the training was made using state observations.

## Abstract

We first launched the agent in the full 3D environment. But the environment was too challenging for it to solve to be able to train.
We then tried to train the training by projecting the 3D space in  2D (The agent was still be reading x,y,z axes but the z values remains 0 for all mobile objects in the environment). We notice then that the agent was training the same way as it does in the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment.
We shifted the trained agent in the 2D fashion in the full 3D environment (from z = 0 to  z = 24), the agent failed mastering the 3D environment.
We shifted the trained agent in 2D to a tiny 3D envirnment (from z = 0 to z = 4) and notice that after losing performance a bit the agent was able to adapt and training in the reduced 3D dimension. 
We decided to shift again by adding 4 units to the z axis (from z= 4 to z = 8) the agent was still adapting. We continued the process untill the full depth (z = 24) and the agent was finally able to play in the full game space and able to last the maximim number of timesteps of an episode. Hence the incremental fashion that made a PPO agent to solve a 3D versio of the slime volley game.

## Key Elements

In the training [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) there are 3 main attributes that make the incremental training a success:
#### Incrementation Threshold
```python
UPGR_LEN  # The performance threshold in terms of mean episode legnth
```
Waiting for the agent to reach the optimal policy before adding more difficulty is time consuming. We set a threshold performance value such that if the agent reaches that threshold we can move it in a upper level of difficulty. We used the average time it lasts in the game as the performance indicator. One would prefer to use the reward as a performance indicator.
#### Incrementation Step Size
```python
n_update # parameter fed to the function setup() of the WORLD class
```
Defines the number of time the z axis will be incremented during the training if initially z = 0. It helps to compute the *incremetatal step* = 24/ n_update. For example if n_update = 6, step  = 4. This means every time the threshold performance is reached the z axis will be increased by 4 units. Note that if the initial depth is greater than 0 the number of incrementations may be less than *n_update*.
#### Training Initial Depth (Z axis)
```python
 init_depth # parameter fed to the function setup() of the WORLD class
```
Precises the value of the z axis at the initialization. Though we recommand a value of 0 to motivate early training, it can also be greater than 0. The maximum initial value we tried that the agent was able to train is 8.

### Environment Setup
An overview of the environment preset before the taining.

```python
env = VolleyBotSelfPlayEnv()
env.training = True # Defaulty False
env.update = True # Defautly to False
env.world.stuck = False # Defautly to False
env.world.setup( n_update = 6, init_depth = 4)                                               
env.seed(SEED)
```
1. *training*: If True the ball will always be launched on the side of the learner, if False the ball will be launch randomnly on both directions. Setting it to True speeds up the training time.
2. *update*: If True the model will be training in a incremental fashion, if False no incrementation training, the model will be train in the full 3D space.
3. *stuck*: If True the initial depth will stay static during the whole training. It's defaulty to False.
4. *world.setup( n_update = 6, init_depth = 4)* : setting up the environment structure before the taining.

## Trained Models

We used the stablebaselines [PPO2](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2) as the training model. But the environment is totally independent of the training algorithm and framework as long as *gym* is installed. You can use Stablebaselines [PPO1](https://github.com/hill-a/stable-baselines/tree/master/stable_baselines/ppo1) or Stablebaselines3 [PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) as well or any other RL or Non RL methods.

The main objectif was to explore *incremental learning* of Deep RL agents not on the used algorithm.

