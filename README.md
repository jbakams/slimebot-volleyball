# slimebot-volleyball: 

https://user-images.githubusercontent.com/59349943/188281632-5513d5a1-447d-4737-9fc8-089ef29249ef.mp4
<p align="center">
  <em>PPO agent trained in an incremental fashion using self-play training(both players are controlled by the same Model)</em>
</p>

slimebot-volleyball is a 3d version of the slime volley game. The game is built on top of [slimevolleygym](https://github.com/hardmaru/slimevolleygym) and uses [Webots](https://cyberbotics.com/) as a simulator for visualization. The major difference with the slimevolleygym environment is the additional z-axis which makes the training more challenging for RL agents such as PPO.

The pretrain PPO in this repository was trained with an Incremental Setting. See details in this [page](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD)

## Presets

In our knowledge just few dependencies are required to run this environment: 

#### 1. Webots:

Webots is open soure and multi-platform software used to simulate robots. It is friendly and doesn't have strong device requirements. [Download](https://cyberbotics.com/) it from the official website and [Install](https://cyberbotics.com/doc/guide/installing-webots) it following the official guidline. We recommand the version R2021b to avoid compatibility bugs with more recent versions.

#### 2. Python:

Defautly Webots use the operating system python. However using a virtual environment is more convinient. Though the environment is fine with any pyhton 3.X, we recommand python 3.7. If using a Webots version different from R2021b, please refer to the [Webots User Guide](https://cyberbotics.com/doc/guide/using-python) to set up python. Otherwise the following [steps](https://github.com/jbakambana/slimebot-volleyball/blob/main/SETTING%20UP.md) explain how to easily set up python in Webots R2021b.

## The Environment

Though Webots has built-in physics system, we used the same physics as with the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) game. This allows to run the environment without Webots for training speeds for example. The trainer is the yellow player and the blue player is the opoponent.

- Observation_Space: Though agents have cameras showing their respective views of the environment and the opponent, the training uses state observation. The pixels version is not yet set up. At each timestep the agent reads the location and speed xyz-coordinates of the ball, the opponent and itself. Which gives a total of 18 inputs for the obsevation.
- Action_space: The basic actions taken by the agent are: *left, right, up, forward, backward  and stay still*. 3 actions maximum can be combined which  gives a total of 18 possible actions.
- The environment can be be dynamic in the *z-axis*, it can take initially any value between 0 and 24 and can change during the training. See this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.
- The objects in the 3D scene was created with [Blender](https://www.blender.org/) (for flexibility) and imported in Webots.

## 3. Training Overview

For now we have able to train a PPO to play the game using self-play and Incremental learnings. Depending on the initialization, the agent can start palying in the full 3D space and last the maximum episode legnth before 10 Million timesteps of training. see the [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) and this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.

Though peforming well, the pretrain agent didn't reach the same perfection in 3D as it did in the 2D version of the game. The actual [champion](link) is the best we got after trying different training settings and seedings. It can be replaced by any other model which beats it during [evaluation](link).

### 3.1. Incrementing sparsity of the training environement

To make it work, we trained the agent in the 3 dimensions environment by stucking the z-axis (meaning the agent will read the 3 axes but the z value of both location and speed will always be zero). We noticed that the agent was training correctly as in 2d [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment.

We realized that by starting in a low level of sparsity and incrementing it progressively as the agent is getting used to the environment was very helpful. So we did the following:

  - The scene has a depth (z axis) of 24 units,we deivided it by 4 and start training the agent in a scene of depth equals to 6 units. To maximize its opportunity to hit the ball.
  
  <p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/simulation_5.gif"></img>
  </p>
  <p align="center">
  <em>The agent training with a narrowed depth at the beggining. Notice that he started to follow the ball</em>
  </p>
  
  - After seeing that the agent has started to master the environment and reach of particular performance (mean reward or mean episode length) we upgrade the depth (let say 12 units) and continue the training. We noticed that,after increasing the depth, the performance will downgrade a bit before starting to go up again. But, at least the agent was able to follow the ball everywhere but was just missing it sometimes.
  - We repeated the same rule of upgration of the depth until the agent will start playting on the maximum depth of the scene meaning 24 units.
  - Depending on the initialization of the Neural Network, in one of the experiment the agent was able to play almost perfectly the game in more or less 10 millions timesteps (sometimes it needs above 20 M timesteps)



## 4. Coming next

### 4.1. Using Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single enviromment to an team play environment (which will increase the observation size). This is why using sensors such as Camera would be a better approach. The input dimension will never change at any environment the agent would  be placed.

- Camera: We would like to use the Webots camera to allow agent to use pixels and see if it can train in both single and teamplay environments.
- Distance sensor: We would like also to see if the built-in distance sensor my be helpul to train the agent as it can detect objects by distance.
- Combine sensors: Human don't use only one sens when performing tasks. A player can react in what he sees and hears from it teammates. We would like to see how to combine different sensors inputs for an optimal training.

### 4.1. Team play
Team not yet successful trained in 3D

https://user-images.githubusercontent.com/59349943/191286958-cacc7b9d-372b-4b66-b275-25181bca6d59.mp4

<p align="center">
  <em>Collaborative selfplay training of 2 PPO in 2D fashion</em>
</p>

## Cocnlusion

This is a lot of statements and questions to answers. Not sure everything will be done in a Master! But let see how far we'll go with the avialable time.



