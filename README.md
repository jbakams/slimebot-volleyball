# slimebot-volleyball: 

Slimebot-volleyball is a 3D version of the slime volleyball game. The game is implemented on top of [slimevolleygym](https://github.com/hardmaru/slimevolleygym) and uses [Webots](https://cyberbotics.com/) as a simulator for visualization. The major difference with the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment is the additional z-axis, which makes the training more challenging for RL agents such as PPO.

---
<p align="left">
  <img width="65%" src="https://github.com/jbakams/slimebot-volleyball/blob/main/Images/mutti-agent_evaluation.gif"></img>
</p>

Evaluation of a team collaboration with 2 independent PPO agents trained in a decentralized fashion with Incrmental Reinforcement Learning and Self-Play Learning. The training script is given [here](https://github.com/jbakams/slimebot-volleyball/blob/main/slimebot-volleyball/2_vs_2/controllers/selfplay_mappo2/selfplay_mappo2.py).

---

## Requirements

#### 1. Webots R2021b:

[Webots](https://cyberbotics.com/)  is open-source and multi-platform software used to simulate robots. It is friendly and doesn't have strong device requirements. [Install](https://cyberbotics.com/doc/guide/installing-webots) it if you are interested in running simulations (Agents can be trained without simulator). We recommand the version R2021b to avoid bugs with recent versions.

#### 2. Python 3.X:

Defaultly, Webots uses the operating system python. However, we recommend using a virtual environment. We used python 3.7 because of Stablebaselines. If  using a Webots version different from R2021b, please refer to the [Webots User Guide](https://cyberbotics.com/doc/guide/using-python) to set up python. Otherwise, use the following [steps](https://github.com/jbakambana/slimebot-volleyball/blob/main/SETTING%20UP.md) to easily set up python in Webots R2021b.

### 3. [Gym](https://github.com/openai/gym)

Slimebot Volleyball is a gym-like envirnment. So one needs to install gym in python.

## The Environment

Although Webots has a built-in physics system, we used the same physics as with the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) game. This allows to run the environment in a CMD without Webots. The training side is the yellow and the blue side is the opponent.

The environment can be dynamic along the *z-axis* (the depth). It can take an initial value between 0 and 24 and can change during the training. See this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.
The objects in the 3D scene were created with [Blender](https://www.blender.org/) (for flexibility) and imported into Webots.

#### 1. Observation space

Although agents have cameras showing their respective views of the environment, We trained agents using *state observation*. The pixels version is not yet set up. At each timestep the agent reads the location and speed XYZ-coordinates of the ball and each agent present in the scene. Depending on the scenario, the observation_space is $(n+1)\times 6$, with $n$ number of agents in the game.

- 1 vs 1: The input shape is 18
- 2 vs 2: The input shape is 30

#### 2. Action Space:

The are 6 basic actions represented with the following 1 hot encding:

- forward    : [1,0,0,0,0]
- backward   : [0,1,0,0,0]
- up         : [0,0,1,0,0]
- right      : [0,0,0,1,0]
- left       : [0,0,0,0,1]
- stay still : [0,0,0,0,0]

3 actions maximum can be combined as long as they don't contradict each other(combining left and right doesn't make sense). Which gives a total of 18 possible actions.

## Scenarios
We have two secnarios at the moment:

#### 1. [1 vs 1](https://github.com/jbakams/slimebot-volleyball/tree/main/slimebot-volleyball/1_vs_1)

<p align="left">
  <img width="65%" src="https://github.com/jbakams/slimebot-volleyball/blob/main/Images/single.png"></img>
</p>

The yellow player represents the agent we aim to training to defeat the blue agent. Cameras at each bottom corners shows views for respective agents.

#### 2. [2 vs 2](https://github.com/jbakams/slimebot-volleyball/tree/main/slimebot-volleyball/2_vs_2)

<p align="left">
  <img width="65%" src="https://github.com/jbakams/slimebot-volleyball/blob/main/Images/team.png"></img>
</p>

The yellow agents represen the team we aim to train to defeat the blue agents. Teammates can collide and bounce if hitting each others. The training is more hectic comparing to the 1 vs 1 case.


## Training Overview

For now, we have been able to train a PPO agent to play the game using *self-play* and *incremental* learning. Depending on the initialization, the agent can start playing in the full 3D space and last the maximum episode length before 10 Million timesteps of training. see the [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/1_vs_1/controllers/selfplay_training_ppo/selfplay_training_ppo.py) and this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.

Though performing well, the pretrained agent didn't reach the same perfection in 3D as it did in the 2D version of the game. The actual [champion]([link](https://github.com/jbakambana/slimebot-volleyball/tree/main/slimebot-volleyball/1_vs_1/controllers/trained_models/ppo2_selfplay)) is the best we got after trying different training settings and seedings. It can be replaced by any other model which beats it during [evaluation](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/evaluation/evaluation.py).


## Coming Next

#### Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single environment to a team play environment (which will increase the observation size). This is why using sensors such as a Camera would be a better approach. The input dimension will never change in any environment in which the agent would be placed.

- Camera: A camera placed on the bot can allow the agent to use pixels as an observation based on its personal view of the 3D environment.
- Distance sensor: The built-in distance sensor may be used to generate observations as the agent can detect objects by distance.
- Combine sensors: Refers to Human abilities to use different senses at the same time. Combining different sensors can bring better training or in the worse case brings confusion to the agent during training.

## Citing the project
```latex
@misc{slimebot-volleyball,
  author = {Bakambana, Jeremie},
  title = {Slimebot Volleyball: A 3D multi-agents gym environment for the slime volleyball game},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jbakams/slimebot-volleyball}},
}
```
## Conclusion

The code might miss some elegance in its structure and syntaxe. Any contribution to improve the the project is welcome. Found another training methods than the Incremental Learning? We can still add it here.


