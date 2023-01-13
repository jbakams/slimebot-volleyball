---
https://user-images.githubusercontent.com/59349943/204865773-d5ba24e8-9ec2-4d5d-8427-6a8fa52e2abe.mp4




Evaluation of two PPO agents trained with Incrmental Reinforcement Learning. Both agents were trained separately using Self-Play training. The yellow agent has been trained with and incremental step of 1 and the blue agent has been trained with an incremtal step of 12. More details are given in this [page](https://github.com/jbakams/slimebot-volleyball/blob/main/INCREMENTAL%20TRAINING.MD).

---

## controllers:

The directory contains different scripts and controllers to train agents and control robots in the simulator. [Webots](https://cyberbotics.com) recognizes only controllers created inside the software. If you a controller is created outside of the software it can only use it with other IDEs.

## environments:

Contains the environment script.

## helpers:

Contains some functional scripts such as the callback class

## worlds:

Contains different game scene and all 3D objects. The file can be open from Webots via the command "File>Open World".
