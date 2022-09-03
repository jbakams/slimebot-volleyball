# slimebot-volleyball: 

<p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/vid1.mkv"></img>
</p>
<p align="center">
  <em>Self-play training of a PPO agent (both players are controlled by the same agent)</em>
</p>


3D-volleyball it's a 3d versions of the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) using [Webots](https://cyberbotics.com/) for the visualization. We added a dimension in the slimevolleygym environment and kept the same settings to see if agents will still be able to learn to the game.

**Notes:** We apologize for the:
-  Poor design of the game, we are working on it. We suppose the game area is surrounded by a wall that we kept invisble for visualization.
-  Missing code in the repository
-  Spelling at the moment

# Installation

## 1. Webots

Webots is open soure and multi-platform software used to simulate robots. It is friendly and doesn't have strong device requirements. [Download](https://cyberbotics.com/) it from the official website and [Install](https://cyberbotics.com/doc/guide/installing-webots) it following the official guidline. We recommand the version R2021b to avoid compatibility bugs with more recent versions.

## 2. Python

Defautly Webots use the system python. However using a virtual env created via [venv](https://docs.python.org/3/library/venv.html) or [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is more convinient. Though the environment is fine with anyt pyhton 3.X, we recommand python 3.7. If using a Webots version different from R2021b, please refer to the [Webots User Guide](https://cyberbotics.com/doc/guide/using-python) to set up python. Otherwise, use the following [steps](https://cyberbotics.com/doc/guide/using-python) to set up python in R2021b.

## 3. Training of the single agent

The image above shows the traing of a  [OpenAi baselines PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) agent using state observation which includes: location and speed xyz-coordinates of the ball, the opponent and itself (18 input size). The same neural network is playing both sides but uses the experience of the yellow player to update its policy.

We were launching the ball on the side of the trainer (yellow player) but after 10 M timesteps no imporvement has been shown. The reason is that, as the ball is always launched in randomly in the 3 axes the ball was practically not hitting the agent to make it understanding how good it is to hit the ball.

<p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/simulation_2.gif"></img>
</p>
<p align="center">
  <em>Training the agent with the full depth at the beggining (Notice the ball is always launched on the side of the learner)</em>
</p>



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

### 3.2. Problem to solve

After this experiment some questions is left to us to investigate in. As we noticed that the 3d version is training by a low level of complexity of the state environment. What is the better way to upgrade the environment when the agent starts to play correctly the game:

-  Start with a depth of zero  and increment the depth gradually;
-  Start with a small non zero depth and increment the depth gradually;
-  After the initial depth, should we go to short incrematation or we can directly jump to the maximum depth
-  How to explain that scientifically

## 4. The Main Goals

### 4.1. Team play

<p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/team.png"></img>
  </p>
  <p align="center">
  <em>Team play volleyball</em>
  </p>

We are setup up a multi-agent team play environment to run some basic experiments such as:

- The whole team managed by the same neural network;
- Each teammate will be an independent NN
- See the possibility of running Multi-Agent RL algorithms: QMix, MAD4PG, etc.
- See if a NN trained in a single environment can learn to collaborate in team-play environment
- And all other ideas that will pop up in our minds

### 4.2. Using Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single enviromment to an team play environment (which will increase the observation size). This is why using sensor such as Camera would be a better approach. The input dimension will never change at any environment the agent would  be placed.

- Camera: We would like to use the Webots camera to allow agent to use pixels and see if it can train in both single and teamplay environments.
- Distance sensor: We would like also to see if the built-in distance sensor my be helpul to train the agent as it can detect objects by distance.
- Combine sensors: Human don't use only one sens when performing tasks. A player can react in what he sees and hears from it teammates. We would like to see how to combine different sensors inputs for an optimal training.

## Conclusion

This is a lot of statements and questions to answers. Not sure everything will be done in a Master! But let see how far we'll go with the avialable time.



