# 3D-volleyball: 

<p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/simulation.gif"></img>
</p>

3D-volleyball it's a 3d versions of the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) using [Webots](https://cyberbotics.com/) for the visualization. We added a dimension in the slimevolleygym environment and kept the same settings to see if agents will still be able to learn to master the game.

# Installation

## 1. Webots

Webots is open soure and multi-platform software used to simulate robots. It is friendly and doesn't have strong device requirements. [Download](https://cyberbotics.com/) and Install it from its official webesite. 

## 2. Python

Defautly Webots use the system python. However, one would prefer a virtual env create via [venv](https://docs.python.org/3/library/venv.html) or [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

(To be continued ....)

## 3. Training of the single agent

The image above shows the traing of a  [OpenAi baselines PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) agent using state observation which includes: location and speed xyz-coordinates of the ball, the opponent and itself (18 input size). The agent is playing both side but uses the experience of the yellow player to update its policy.

We were launching the ball on the side of the trainer (yellow player) but after 10 M timesteps no imporvement has been shown. The reason is that, as the ball is always launched in randomly in the 3 axes the ball was practically not hitting the agent to make it understanding how good it is to hit the ball.

<p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/simulation_5.gif"></img>
</p>



## 3.1. Incrementing sparsity of the training environement

To make it work, we trained the agent in the 3 dimensions environment by stucking the z-axis (meaning the agent will read the 3 axes but the z value of both location and speed will always be zero). We noticed that the agent was training correctly as in 2d [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment.

We realized that by starting in a low level of sparsity and incrementing it progressively as the agent is getting used to the environment was very helpful. So we did the following:

  - The scene has a depth (z axis) of 24 units,we deivided it by 4 and start training the agent in a scene of depth equals to 6 units. To maximize its opportunity to hit the ball.
  
  <p align="center">
  <img width="75%" src="https://github.com/jbakambana/3D-slimevolley/blob/main/Images/simulation_2.gif"></img>
  </p>
  
  - After seeing that the agent has started to master the environment and reach of particular performance (mean reward or mean episode length) we upgrade the depth (let say 12 units) and continue the training. We noticed that,after increasing the depth, the performance will downgrade a bit before starting to go up again. But, at least the agent was able to follow the ball everywhere but was just missing it sometimes.
  - We repeated the same rule of upgration of the depth until the agent will start playting on the maximum depth of the scene meaning 24 units.
  - Depending on the initialization of the Neural Network, in one of the experiment the agent was able to play almost perfectly the game in more or less 10 millions timesteps (sometimes it needs above 20 M timesteps)

## 3.2. Problem to solve

After this experiment some questions is left to us to investigate in. As we noticed that the 3d version is training by a low level of complexity of the state environment. What is the better way to upgrade the environment when the agent starts to play correctly the game:

-  Start with a depth of zero  and increment the depth gradually;
-  Start with a small non zero depth and increment the depth gradually;
-  After the initial depth, should we go to short incrematation or we can directly jump to the maximum depth
-  How to explain that scientifically

## 4. The Main Goal

### 4.1. Team playing

We are setup up a multi-agent team play environment to run some basic experiments such as:

- The whole team managed by the same neural network;
- Each teammate will be an independent NN
- See the possibility of runnung Multi-Agent RL algorithms: QMix, MAD4PG, etc.
- See if a NN trained in a single environment can learn to collaborate in team-play environment
- And all other ideas that will pop up in our minds

### 4.2. Use Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single enviromment to an team play environment (which will increase the observation size). This is why using sensor such as Camera would be a better approach. The input dimension will never change at any environment the agent would  be placed.

- Camera: We would like to use the Webots camera to allow agent to use pixels and see if it can train in both single and teamplay environments.
- Distance sensor: We would like also to see if the built-in distance sensor my be helpul to train the agent as it can detect objects by distance.
- Combine sensors: Human don't use only one sens when performing tasks. A player can react in what he sees and hears from it teammates. We would like to see how to combine different sensors inputs for an optimal training.

## Conclusion

This is a lot of statements and questions to answers. Not sure everything will be done in a Master! But let see how far we'll go with the avialable time.



