This post explain how we managed to train the PPO algorithm to solve the game with Incremental Reinfrocement Learning.

# Self-Play Incremental Training

We aimed to train the PPO from scratch in the 3D environment but the algorithm did not show progress after more than twenty million timesteps. As shown in the video below, we applied Incremental Learning, by freezing the $z-$axis to made agents and ball moving only along x and y axes, similarly to the slimevolleygym environment. We defined an Performance Threshold such that every time the mean episode length surpass the Threshold, the environment get incremented along the z-axis according a given incremental step. We repeat the same process untill the environment reaches a maximum depth of 24.

https://user-images.githubusercontent.com/59349943/191252794-0a18fbee-15ed-49fb-9d7b-c1c50f2d4ea8.mp4

<p align="center">
  <em>Top view of the incremental learning process with an Initial Depth of 0, Threshold of 1500 and Incremental Step of 4</em>
</p>

In the follwing Figures we can see two different incremental training. The first graphic show the incremental training progress with a low threshold and a small incremental step, and the second one with a high threshold and an big incremental step.

---

![graph_500](https://user-images.githubusercontent.com/59349943/204380885-f3d1f0b6-cb66-4ba9-a453-0d2e1a63390e.png)
  
<p align="center">
  <em>Illustration of incremental learning with an initial depth of 0 (z = 0), performance threshold of 500, and uncremental step of 1. This means every time the average episode length reaches 500 the depth gets incremented for 1 unit. By counting the red diamonds, we see that the PPO adapted to 24 increments</em>
</p>

---

![graph_2500](https://user-images.githubusercontent.com/59349943/204382042-db0b95e5-58f6-49b3-8dc8-4e9ee3bb4ce9.png)

<p align="center">
  <em>Illustration of incremental learning with an initial depth of 0 (z = 0), performance threshold of 2500, and incremental step of 12. In this case every time the average episode length reaches 2500 the depth gets incremented for 12 units. This means the environment will increment only twice to reach a depth of 24. And we see that the PPO algorithm adapted to the increment even if the incremental step was big.
</p>
  
---

 The yellow agent is the main page is the one trained with a threshold of 500 and incremental step of 1; the Blue player is the one trained with a threshold of 2500 and incremental step of 12. We can see in the main page video that both agent expertises are similar even if they didn't train gradually the same way.

## Key Elements

In the training [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) 3 main attributes make the incremental training a success:
#### Incrementation Threshold
```python
INCREMENT_THRESHOLD  # The performance threshold in terms of mean episode length
```
Waiting for the agent to reach the optimal policy before adding more difficulty is time-consuming. We set a threshold performance value such that if the agent reaches that threshold we can move it to an upper level of difficulty. We used the average time it lasts in the game as the performance indicator. One would prefer to use the reward as a performance indicator.
#### Incrementation Step Size
```python
n_increment # parameter fed to the function setup() of the WORLD class
```
Defines the number of times the z-axis will be incremented during the training if initially z = 0. It helps to compute the *incremental step* = 24/ n_update. For example if n_increment = 6, step  = 4. This means every time the threshold performance is reached the z-axis will be increased by 4 units. Note that if the initial depth is greater than 0 the number of incrementations may be less than *n_update*.
#### Training Initial Depth (Z axis)
```python
 init_depth # parameter fed to the function setup() of the WORLD class
```
Precise the value of the z-axis at the initialization. Though we recommand a value of 0 to motivate early training, it can also be greater than 0. The maximum initial value we tried that the agent was able to train is 8.

### Environment Setup
An overview of the environment preset before the training.

```python
env = VolleyBotSelfPlayEnv()
env.training = True # Defaulty False
env.update = True # Defautly to False
env.world.stuck = False # Defautly to False
env.world.setup( n_increment = 6, init_depth = 4)                                               
env.seed(SEED)
```
1. *training*: If True the ball will always be launched on the side of the learner if False the ball will be launched randomly in both directions. Setting it to True speeds up the training time.
2. *update*: If True the model will be trained incrementally, if False no incrementation training, the model will be trained in the full 3D space.
3. *stuck*: If True the initial depth will stay static during the whole training. It defaults to False.
4. *world.setup( n_increment = 6, init_depth = 4)* : setting up the environment structure before the training.

---

## Trained Model

We used the stablebaselines [PPO2](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2) as the training model. But the environment is independent of the training algorithm and framework as long as *gym* is installed. You can use Stablebaselines [PPO1](https://github.com/hill-a/stable-baselines/tree/master/stable_baselines/ppo1) or Stablebaselines3 [PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) as well or any other RL or Non-RL methods.

The main objective was to explore *incremental learning* of Deep RL agents not on the used algorithm.
