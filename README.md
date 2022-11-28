# slimebot-volleyball: 

https://user-images.githubusercontent.com/59349943/204383877-084e76f4-82da-4c67-9a5b-cb33832bf546.mp4

<p align="center">
  <em>PPO agents trained with Incrmental Reinforcement Learning. Both agents were trained separately using Self-Play training. The yellow agent has been trained with and incremental step of 1 and the blue agent has been trained with an incremtal step of 12 </em>
</p>

---

Slimebot-volleyball is a 3d version of the slime volley game. The game is built on top of [slimevolleygym](https://github.com/hardmaru/slimevolleygym) and uses [Webots](https://cyberbotics.com/) as a simulator for visualization. The major difference with the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment is the additional z-axis which makes the training more challenging for RL agents such as PPO.

The pre-trained PPO in this repository was trained with an Incremental Fashion. See details in this [page](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD)

---

## Presets

In our knowledge just a few dependencies are required to run this environment: [gym](https://github.com/openai/gym), open_cv

#### 1. Webots:

Webots is open-source and multi-platform software used to simulate robots. It is friendly and doesn't have strong device requirements. [Download](https://cyberbotics.com/) it from the official website and [Install](https://cyberbotics.com/doc/guide/installing-webots) it following the official guidelines. We recommand version R2021b to avoid compatibility bugs with more recent versions.

#### 2. Python:

Default Webots use the operating system python. However, using a virtual environment is more convenient. Though the environment is fine with any python 3.X, we recommand python 3.7. If using a Webots version different from R2021b, please refer to the [Webots User Guide](https://cyberbotics.com/doc/guide/using-python) to set up python. Otherwise, the following [steps](https://github.com/jbakambana/slimebot-volleyball/blob/main/SETTING%20UP.md) explain how to easily set up python in Webots R2021b.

## The Environment

Though Webots has a built-in physics system, we used the same physics as with the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) game. This allows running the environment without Webots for training speeds for example. The trainer is the yellow player and the blue player is the opponent.

- Observation_Space: Though agents have cameras showing their respective views of the environment and the opponent, the training uses state observation. The pixels version is not yet set up. At each timestep the agent reads the location and speed XYZ-coordinates of the ball, the opponent, and itself. Which gives a total of 18 inputs for the observation.
- Action_space: The basic actions taken by the agent are: *left, right, up, forward, backward, and stay still*. 3 actions maximum can be combined which gives a total of 18 possible actions.
- The environment can be dynamic in the *z-axis*, it can take initially any value between 0 and 24 and can change during the training. See this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.
- The objects in the 3D scene were created with [Blender](https://www.blender.org/) (for flexibility) and imported into Webots.

## Training Overview

For now, we have been able to train a PPO agent to play the game using *self-play* and *incremental* learning. Depending on the initialization, the agent can start playing in the full 3D space and last the maximum episode length before 10 Million timesteps of training. see the [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) and this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.

Though performing well, the pretrained agent didn't reach the same perfection in 3D as it did in the 2D version of the game. The actual [champion]([link](https://github.com/jbakambana/slimebot-volleyball/tree/main/slimebot-volleyball/controllers/trained_models/ppo2_selfplay)) is the best we got after trying different training settings and seedings. It can be replaced by any other model which beats it during [evaluation](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/evaluation/evaluation.py).


## Challenges

#### Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single environment to a team play environment (which will increase the observation size). This is why using sensors such as a Camera would be a better approach. The input dimension will never change in any environment in which the agent would be placed.

- Camera: A camera placed on the bot can allow the agent to use pixels as an observation based on its personal view of the 3D environment.
- Distance sensor: The built-in distance sensor may be used to generate observations as the agent can detect objects by distance.
- Combine sensors: Refers to Human abilities to use different senses at the same time. Combining different sensors can bring better training or in the worse case brings confusion to the agent during training.

#### Collaborative Team play
The team not yet successful trained in 3D

https://user-images.githubusercontent.com/59349943/191286958-cacc7b9d-372b-4b66-b275-25181bca6d59.mp4

<p align="center">
  <em>Collaborative selfplay training of 2 PPO in 2D fashion</em>
</p>

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


