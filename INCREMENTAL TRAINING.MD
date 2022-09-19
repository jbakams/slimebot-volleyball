# Incremental Training

The following training method is a part of my research Master Project at Stellenbosch Unversity. Though the agent has camera, the training was made using state observation.

## Abstract

We first launched the agent in the full 3D environment. But the environment was too challenging for it to solve to be able to train.
We then tried to train the training by projecting the 3D space in  2D (The agent was still be reading x,y,z axes but the z values remains 0 for all mobile objects in the environment). We notice then that the agent was training the same way as it does in the [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment.
We shifted the trained agent in the 2D fashion in the full 3D environment (from z = 0 to  z = 24), the agent failed mastering the 3D environment.
We shifted the trained agent in 2D to a tiny 3D envirnment (from z = 0 to z = 4) and notice that after losing performance a bit the agent was able to adapt and training in the reduced 3D dimension. 
We decided to shift again by adding 4 units to the z axis (from z= 4 to z = 8) the agent was still adapting. We continued the process untill the full depth (z = 24) and the agent was finally able to play in the full game space and able to last the maximim number of timesteps of an episode. Hence the incremental fashion that made a PPO agent to solve a 3D versio of the slime volley game.

## Scripts

### The environment

We jsut expalining the main attribute of the environment that help in the incremental setting.

```python
env = VolleyBotEnv()

# training = True means during training the ball will always be launched on the learning, this speeds up the training time
env.training = True 

# update = True, means the z axis will be incremented each time the performance threshold is reaching during evaluation
env.update = True 

# stuck = True, means no incrementation learning, the initial depth will stay fixed during the whole training
env.world.stuck = False 

# n_update precises the value of the step at each incrementation, step = 24/n_update
# init_depth precises the value of the z axis at the initialization, it's not oblige to start with 0
env.world.setup( n_update = 4, init_depth = 8) 
                                              
env.seed(SEED)
```
