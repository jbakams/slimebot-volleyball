*The following script is a part of my MEng research project. Feel free to contact me for more details.*

# Self Play Incremental Training

The *tabula rasa* failed in training the RL agent in the 3D envirnment. We try to train the agent from scratch in the 3D environment but the agent didn't show any progress at all after more than 20M timesteps. As the agent is able to train in the 2D version, we explored the ability of training it using a curriculum incremental fashion from 2D to 3D which worked hopefully.

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

