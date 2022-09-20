# slimebot-volleyball: 

https://user-images.githubusercontent.com/59349943/188281632-5513d5a1-447d-4737-9fc8-089ef29249ef.mp4
<p align="center">
  <em>PPO agent trained in an incremental fashion using self-play training(both players are controlled by the same Model)</em>
</p>

--

Slimebot-volleyball is a 3d version of the slime volley game. The game is built on top of [slimevolleygym](https://github.com/hardmaru/slimevolleygym) and uses [Webots](https://cyberbotics.com/) as a simulator for visualization. The major difference with the slimevolleygym environment is the additional z-axis which makes the training more challenging for RL agents such as PPO.

The pre-trained PPO in this repository was trained with an Incremental Setting. See details in this [page](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD)

---

## Presets

In our knowledge just few dependencies are required to run this environment: [gym](https://github.com/openai/gym), open_cv

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

## Training Overview

For now we have able to train a PPO to play the game using self-play and Incremental learnings. Depending on the initialization, the agent can start palying in the full 3D space and last the maximum episode legnth before 10 Million timesteps of training. see the [script](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/selfplay_training_ppo/selfplay_training_ppo.py) and this [post](https://github.com/jbakambana/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD) for more details.

Though peforming well, the pretrain agent didn't reach the same perfection in 3D as it did in the 2D version of the game. The actual [champion]([link](https://github.com/jbakambana/slimebot-volleyball/tree/main/slimebot-volleyball/controllers/trained_models/ppo2_selfplay)) is the best we got after trying different training settings and seedings. It can be replaced by any other model which beats it during [evaluation](https://github.com/jbakambana/slimebot-volleyball/blob/main/slimebot-volleyball/controllers/evaluation/evaluation.py).


## Challenges

#### Webots built-in sensors

The use of state observation shows weakness in the input dimension when we need to move an agent trained in a single enviromment to an team play environment (which will increase the observation size). This is why using sensors such as Camera would be a better approach. The input dimension will never change at any environment the agent would  be placed.

- Camera: A camera placed on the bot can allow agent to use pixels as observation based on its personal view of the 3D environment.
- Distance sensor: The built-in distance sensor may be used to generate observations as the agent can detect objects by distance.
- Combine sensors: Refering on Human abilities of using different sens at the same time. Combining different sensors can bring better training or in the worse case brings confusion to the agent during training.

#### Collaborative Team play
Team not yet successful trained in 3D

https://user-images.githubusercontent.com/59349943/191286958-cacc7b9d-372b-4b66-b275-25181bca6d59.mp4

<p align="center">
  <em>Collaborative selfplay training of 2 PPO in 2D fashion</em>
</p>

## Citing the project
```latex
@misc{slimebot volleyball: A multi-agents 3D gym environment for slime volleyball,
  author = {Bakambana, Jeremie},
  title = {Slimebot Volleyball},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jbakambana/slimebot-volleyball}},
}
```
## Conclusion

The code might miss some elegance in its structure and syntaxe. Any contribution to improve the the project is welcome. Found another training methods than the Incremental Learning? We can still add it here.


