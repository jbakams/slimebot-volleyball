"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import cv2 # installed with gym anyways
from collections import deque
from time import sleep
from controller import Supervisor ## Webots method to control nodes in the 3D scene area


np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5
MAXLIVES = 5 # game ends when one agent loses this many games

### Webots set up
supervisor = Supervisor()
TIME_STEP = 32

class World:

  def __init__(self, upgrade = True, z_axis = True):
  
    
    self.width = 24*2
    self.height = self.width
    self.max_depth = self.width/2
    self.step = -np.inf
    self.depth = -np.inf  
      
    
    self.wall_width = 1.0 # wall width
    self.wall_height = 2.0
    self.wall_depth = -np.inf
    self.player_vx = 10*1.75
    self.player_vy = 10*1.35
    self.player_vz = 10*1.75
    self.max_ball_v = 15*1.5   
    self.z_axis = z_axis 
    
    self.gravity = -9.8*2*1.5   
    self.upgrade = upgrade
    self.setup()
  
  def setup(self, n_upgr = 4, init_depth = 6):
    if not self.upgrade:
      self.wall_depth = self.depth = self.max_depth 
      
    else: 
        self.step = self.max_depth/n_upgr  
        self.wall_depth  = self.depth = init_depth  

    
  def upgrade_world(self):
    
    if self.upgrade:
      
      ball_speed_ratio = self.max_ball_v/self.max_depth    

      self.depth +=  self.step
      self.wall_depth = self.depth      
      
    
    if self.depth >= self.max_depth:
      self.depth = self.max_depth
      self.wall_depth = self.depth                     
          
      self.upgrade = False
      
    
WORLD = World()   
      


class DelayScreen:
  """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """
  def __init__(self, life=INIT_DELAY_FRAMES):
    self.life = 0
    self.reset(life)
  def reset(self, life=INIT_DELAY_FRAMES):
    self.life = life
  def status(self):
    if (self.life == 0):
      return True
    self.life -= 1
    return False


class Particle:
  """ used for the ball, and also for the round stub above the fence """
  def __init__(self, x, y, z, vx, vy, vz, r, name):
    self.x = x
    self.y = y
    self.z = z
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.vx = vx
    self.vy = vy
    self.vz = vz 
    self.r = r
    
    
    self.particule = supervisor.getFromDef(name)
    self.location = self.particule.getField("translation")
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
    
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def applyAcceleration(self, ax, ay, az):
    self.vx += ax * TIMESTEP
    self.vy += ay * TIMESTEP
    self.vz += az * TIMESTEP * (WORLD.depth/WORLD.max_depth)
    
    
    
  def checkEdges(self):
    if (self.x <= (self.r-WORLD.width/2)):
      self.vx *= -FRICTION
      self.x = self.r-WORLD.width/2+NUDGE*TIMESTEP

    if (self.x >= (WORLD.width/2-self.r)):
      self.vx *= -FRICTION;
      self.x = WORLD.width/2-self.r-NUDGE*TIMESTEP
      
    if WORLD.depth >= self.r:
      
        if (self.z<=(self.r-WORLD.depth/2)):
          self.vz *= -FRICTION * (WORLD.depth/WORLD.max_depth)
          self.z = self.r-WORLD.depth/2+NUDGE*TIMESTEP
          
        if (self.z >= (WORLD.depth/2-self.r)):
          self.vz *= -FRICTION * (WORLD.depth/WORLD.max_depth);
          self.z = WORLD.depth/2-self.r-NUDGE*TIMESTEP
          
    else:
        if (self.z<=(WORLD.depth/2)):
          self.vz *= -FRICTION * (WORLD.depth/WORLD.max_depth)
          self.z =-WORLD.depth/2+NUDGE*TIMESTEP
          
        if (self.z >= (WORLD.depth/2)):
          self.vz *= -FRICTION * (WORLD.depth/WORLD.max_depth);
          self.z = WORLD.depth/2-NUDGE*TIMESTEP

    

    if (self.y<=(self.r)):
      self.vy *= -FRICTION
      self.y = self.r+NUDGE*TIMESTEP
      
      self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
      
      if (self.x <= 0):
        return -1
      else:
        return 1
        
    if (self.y >= (WORLD.height-self.r)):
      self.vy *= -FRICTION
      self.y = WORLD.height-self.r-NUDGE*TIMESTEP
      
    # fence:
    if ((self.x <= (WORLD.wall_width/2+self.r)) and (self.prev_x > (WORLD.wall_width/2+self.r)) and (self.y <= WORLD.wall_height)):
      self.vx *= -FRICTION
      self.x = WORLD.wall_width/2+self.r+NUDGE*TIMESTEP

    if ((self.x >= (-WORLD.wall_width/2-self.r)) and (self.prev_x < (-WORLD.wall_width/2-self.r)) and (self.y <= WORLD.wall_height)):
      self.vx *= -FRICTION
      self.x = -WORLD.wall_width/2-self.r-NUDGE*TIMESTEP
      
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    return 0;
    
  def getDist2(self, p): # returns distance squared from p
    dz = p.z - self.z
    dy = p.y - self.y
    dx = p.x - self.x
    
    return (dx*dx+dy*dy+dz*dz)
    
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    if WORLD.depth != 0:
      return (r*r > self.getDist2(p) and (self.z*self.z <= WORLD.wall_depth * WORLD.wall_depth)) # if distance is less than total radius and the depth, then colliding.
    else:
      return (r*r > self.getDist2(p))
      
  def bounce(self, p): # bounce two balls that have collided (this and that)
    abx = self.x-p.x
    aby = self.y-p.y
    abz = self.z-p.z
    abd = math.sqrt(abx*abx+aby*aby+abz*abz)
    abx /= abd # normalize
    aby /= abd
    abz /= abd
    nx = abx # reuse calculation
    ny = aby
    nz = abz
    abx *= NUDGE
    aby *= NUDGE
    abz *= NUDGE
    while(self.isColliding(p)):
      self.x += abx
      self.y += aby
      self.z += abz
      
      self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
      
    ux = self.vx - p.vx
    uy = self.vy - p.vy
    uz = self.vz - p.vz
    un = ux*nx + uy*ny + uz*nz
    unx = nx*(un*2.) # added factor of 2
    uny = ny*(un*2.) # added factor of 2
    unz = nz*(un*2.) # added factor of 2
    ux -= unx
    uy -= uny
    uz -= unz
    self.vx = ux + p.vx
    self.vy = uy + p.vy
    self.vz = (uz + p.vz ) #* (self.world.depth/self.world.max_depth)
    
  
    
  def limitSpeed(self, minSpeed, maxSpeed):
    mag2 = self.vx*self.vx+self.vy*self.vy+self.vz*self.vz;
    if (mag2 > (maxSpeed*maxSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vz /= mag
      self.vx *= maxSpeed
      self.vy *= maxSpeed
      self.vz *= maxSpeed * (WORLD.depth/WORLD.max_depth)

    if (mag2 < (minSpeed*minSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vz /= mag
      self.vx *= minSpeed
      self.vy *= minSpeed
      self.vz *= minSpeed * (WORLD.depth/WORLD.max_depth)

class Wall:
  """ used for the fence, and also the ground """
  def __init__(self, x, y, z, w, h, d):
    self.x = x;
    self.y = y;
    self.z = z;
    self.w = w;
    self.h = h;
    self.d = d;
    
  

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.z = 0
    self.vx = 0
    self.vy = 0
    self.vz = 0
    # ball
    self.bx = 0
    self.by = 0
    self.bz = 0
    self.bvx = 0
    self.bvy = 0
    self.bvz = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.oz = 0
    self.ovx = 0
    self.ovy = 0
    self.ovz = 0
  def getObservation(self):
    result = [self.x, self.y, self.z, self.vx, self.vy, self.vz,
              self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
              self.ox, self.oy, self.oz, self.ovx, self.ovy, self.ovz]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent():
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, dir, x, y, z, name):
    
    self.dir = dir # -1 means left, 1 means right player for symmetry.
    self.x = x
    self.y = y
    self.z = z
    self.r = 1.5
    
    self.vx = 0
    self.vy = 0
    self.vz = 0
    self.desired_vx = 0
    self.desired_vy = 0
    self.desired_vz = 0
    self.state = RelativeState()
    self.emotion = "happy"; # hehe...
    self.life = MAXLIVES
    
    self.agent = supervisor.getFromDef(name)
    self.location = self.agent.getField("translation")
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def lives(self):
    return self.life
    
  def setAction(self, action):
    forward = False
    backward = False
    jump = False
    right = False
    left = False
    
    if action[0] > 0:
      forward = True
    if action[1] > 0:
      backward = True
    if action[2] > 0:
      jump = True
    if action[3] > 0:
      right = True
    if action[4] > 0:
      left = True
        
    self.desired_vx = 0
    self.desired_vy = 0
    self.desired_vz = 0
        
        
    if (forward and (not backward)):
      self.desired_vx = -WORLD.player_vx
    if (backward and (not forward)):
      self.desired_vx = WORLD.player_vx
    if jump:
      self.desired_vy = WORLD.player_vy
    if (right and (not left)):
      self.desired_vz = WORLD.player_vz
    if (left and (not right)):
      self.desired_vz = -WORLD.player_vz
      
  def move(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
    
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def step(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
    
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def update(self):
    self.vy += GRAVITY * TIMESTEP

    if (self.y <=  NUDGE*TIMESTEP):
      self.vy = self.desired_vy

    self.vx = self.desired_vx*self.dir
    self.vz = self.desired_vz

    self.move()

    if (self.y <= 0):
      self.y = 0;
      self.vy = 0;

    # stay in their own half:
    if (self.x*self.dir <= (WORLD.wall_width/2+self.r) ):
      self.vx = 0;
      self.x = self.dir*(WORLD.wall_width/2+self.r)

    if (self.x*self.dir >= (WORLD.width/2-self.r) ):
      self.vx = 0;
      self.x = self.dir*(WORLD.width/2-self.r)
      
    # stay in scene area:
    
    if WORLD.wall_depth >= self.r:
      if (self.z <= -(WORLD.wall_depth/2 - self.r) ):
        self.vz = 0;
        self.z = -(WORLD.wall_depth/2-self.r)
    
      if (self.z >= WORLD.wall_depth/2 - self.r ):
        self.vz = 0;
        self.z = WORLD.wall_depth/2 - self.r
    else:
      if (self.z <= -(WORLD.wall_depth/2) ):
        self.vz = 0;
        self.z = -(WORLD.wall_depth/2)
    
      if (self.z >= WORLD.wall_depth/2):
        self.vz = 0;
        self.z = WORLD.wall_depth/2
     
    self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
  def updateState(self, ball, opponent):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x*self.dir
    self.state.y = self.y
    self.state.z = self.z
    self.state.vx = self.vx*self.dir
    self.state.vy = self.vy
    self.state.vz = self.vz
    # ball
    self.state.bx = ball.x*self.dir
    self.state.by = ball.y
    self.state.bz = ball.z
    self.state.bvx = ball.vx*self.dir
    self.state.bvy = ball.vy
    self.state.bvz = ball.vz
    # opponent
    self.state.ox = opponent.x*(-self.dir)
    self.state.oy = opponent.y
    self.state.oz = opponent.z
    self.state.ovx = opponent.vx*(-self.dir)
    self.state.ovy = opponent.vy
    self.state.ovz = opponent.vz
    
  def getObservation(self):
    return self.state.getObservation()

  

class BaselinePolicy:
  """ Tiny RNN policy with only 120 parameters of otoro.net/slimevolley agent """
  def __init__(self):
    self.nGameInput = 8 # 8 states for agent
    self.nGameOutput = 3 # 3 buttons (forward, backward, jump)
    self.nRecurrentState = 4 # extra recurrent states for feedback.

    self.nOutput = self.nGameOutput+self.nRecurrentState
    self.nInput = self.nGameInput+self.nOutput
    
    # store current inputs and outputs
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)

    """See training details: https://blog.otoro.net/2015/03/28/neural-slime-volleyball/ """
    self.weight = np.array(
      [7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689,
       1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887,
       2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353,
       -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809,
       7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695,
       -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952,
       -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377])

    self.bias = np.array([2.2935,-2.0353,-1.7786,5.4567,-3.6368,3.4996,-0.0685])

    # unflatten weight, convert it into 7x15 matrix.
    self.weight = self.weight.reshape(self.nGameOutput+self.nRecurrentState,
      self.nGameInput+self.nGameOutput+self.nRecurrentState)
  def reset(self):
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)
  def _forward(self):
    self.prevOutputState = self.outputState
    self.outputState = np.tanh(np.dot(self.weight, self.inputState)+self.bias)
  def _setInputState(self, obs):
    # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
    [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
    self.inputState[0:self.nGameInput] = np.array([x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy])
    self.inputState[self.nGameInput:] = self.outputState
  def _getAction(self):
    forward = 0
    backward = 0
    jump = 0
    if (self.outputState[0] > 0.75):
      forward = 1
    if (self.outputState[1] > 0.75):
      backward = 1
    if (self.outputState[2] > 0.75):
      jump = 1
    return [forward, backward, jump]
  def predict(self, obs):
    """ take obs, update rnn state, return action """
    self._setInputState(obs)
    self._forward()
    return self._getAction()

class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, np_random=np.random, training = False):
    self.ball = None
    self.ground = None
    self.fence = None
    self.fenceStub = None
    self.agent_left = None
    self.agent_right = None
    self.delayScreen = None
    self.np_random = np_random
    
    self.training = training
    self.reset()
    
  def reset(self):    
      
    self.fenceStub = Particle(0, WORLD.wall_height, 0, 0, 0, 0, WORLD.wall_width/2, "FENCESTUB");
    if self.training:
      ball_vx = self.np_random.uniform(low=0, high=20)
    else:
      ball_vx = self.np_random.uniform(low=-20, high=20)       
    ball_vy = self.np_random.uniform(low=10, high=25)
    ball_vz = self.np_random.uniform(low=-10, high=10) * (WORLD.depth/WORLD.max_depth)
    self.ball = Particle(0, (WORLD.width/4)-1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
    self.agent_left = Agent(-1, -WORLD.width/4, 1.5, 0, "BLUE")
    self.agent_right = Agent(1, WORLD.width/4, 1.5, 0, "YELLOW")
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)
    self.delayScreen = DelayScreen()
    
  def newMatch(self):
    if self.training:
      ball_vx = self.np_random.uniform(low=0, high=20)
    else:
      ball_vx = self.np_random.uniform(low=-20, high=20)
    ball_vy = self.np_random.uniform(low=10, high=25)
    ball_vz = self.np_random.uniform(low=-10, high=10) * (WORLD.depth/WORLD.max_depth)
    self.ball = Particle(0, (WORLD.width/4)-1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
    self.delayScreen.reset()
    
  def step(self):
    """ main game loop """

    self.betweenGameControl()
    self.agent_left.update()
    self.agent_right.update()

    if self.delayScreen.status():
      self.ball.applyAcceleration(0, WORLD.gravity, 0)
      self.ball.limitSpeed(0, WORLD.max_ball_v)
      self.ball.move()

    if (self.ball.isColliding(self.agent_left)):
      self.ball.bounce(self.agent_left)
    if (self.ball.isColliding(self.agent_right)):
      self.ball.bounce(self.agent_right)
    self.fenceStub.z = self.ball.z
    if (self.ball.isColliding(self.fenceStub)):
      
      self.ball.bounce(self.fenceStub)

    # negated, since we want reward to be from the persepctive of right agent being trained.
    result = -self.ball.checkEdges()

    if (result != 0):
      self.newMatch() # not reset, but after a point is scored
      if result < 0: # baseline agent won
        self.agent_left.emotion = "happy"
        self.agent_right.emotion = "sad"
        self.agent_right.life -= 1
      else:
        self.agent_left.emotion = "sad"
        self.agent_right.emotion = "happy"
        self.agent_left.life -= 1
      return result

    # update internal states (the last thing to do)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)

    return result
  
  def betweenGameControl(self):
    agent = [self.agent_left, self.agent_right]
    if (self.delayScreen.life > 0):
      pass
      '''
      for i in range(2):
        if (agent[i].emotion == "sad"):
          agent[i].setAction([0, 0, 0, 0, 0]) # nothing
      '''
    else:
      agent[0].emotion = "happy"
      agent[1].emotion = "happy"

class Slime3DEnv(gym.Env):
  """
  Gym wrapper for Slime Volley game.

  By default, the agent you are training controls the right agent
  on the right. The agent on the left is controlled by the baseline
  RNN policy.

  Game ends when an agent loses 5 matches (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  # for compatibility with typical atari wrappers
  atari_action_meaning = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
  }
  atari_action_set = {
    0, # NOOP
    4, # LEFT
    7, # UPLEFT
    2, # UP
    6, # UPRIGHT
    3, # RIGHT
  }

  action_table = [[0, 0, 0, 0, 0], # NOOP
                  [1, 0, 0, 0, 0], # FORWARD
                  [1, 0, 1, 0, 0], # FORWARD JUMP
                  [1, 0, 0, 1, 0], # FORWARD RIGHT
                  [1, 0, 0, 0, 1], # FORWARD LEFT
                  [1, 0, 1, 1, 0], # FORWARD JUMP RIGHT
                  [1, 0, 1, 0, 1], # FORWARD JUMP LEFT
                  [0, 1, 0, 0, 0], # BACKWARD
                  [0, 1, 1, 0, 0], # BACKWARD JUMP
                  [0, 1, 0, 1, 0], # BACKWARD RIGHT
                  [0, 1, 0, 0, 1], # BACKWARD LEFT
                  [0, 1, 1, 1, 0], # BACKWARD JUMP RIGHT
                  [0, 1, 1, 0, 1], # BACKWARD JUMP LEFT
                  [0, 0, 1, 0, 0], # JUMP 
                  [0, 0, 1, 1, 0], # JUMP RIGHT
                  [0, 0, 1, 0, 1], # JUMP LEFT
                  [0, 0, 0, 1, 0], # RIGHT
                  [0, 0, 0, 0, 1]] # LEFT

  from_pixels = False
  atari_mode = False
  survival_bonus = False # Depreciated: augment reward, easier to train
  multiagent = True # optional args anyways

  def __init__(self, training = False, upgrade = False, num_upgr = 0, step_upgr = 0):
   
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py

    Setting self.from_pixels to True makes the observation with multiples
    of 84, since usual atari wrappers downsample to 84x84
    """
    
    self.t = 0
    self.t_limit = 3000
    
    self.num_envs = 1
    self.training = training
    self.upgrade = upgrade
    self.world = WORLD

    #self.action_space = spaces.Box(0, 1.0, shape=(3,))
    if self.atari_mode:
      self.action_space = spaces.Discrete(18)
    else:
      self.action_space = spaces.MultiBinary(5)

    if self.from_pixels:
      setPixelObsMode()
      self.observation_space = spaces.Box(low=0, high=255,
        shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
    else:
      high = np.array([np.finfo(np.float32).max] * 18)
      self.observation_space = spaces.Box(-high, high)
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game(training = self.training)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function

    self.policy = BaselinePolicy() # the “bad guy”

    self.viewer = None

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(np_random=self.np_random, training = self.training)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]

  def getObs(self):
    if self.from_pixels:
      obs = self.render(mode='state')
      self.canvas = obs
    else:
      obs = self.game.agent_right.getObservation()
    return obs

  def discreteToBox(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 5:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 18)
    return self.action_table[n]

  def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1
    supervisor.step(TIME_STEP)
    if self.otherAction is not None:
      otherAction = self.otherAction
      
    if otherAction is None: # override baseline policy
      obs = self.game.agent_left.getObservation()
      otherAction = self.policy.predict(obs)

    if self.atari_mode:
      action = self.discreteToBox(action)
      otherAction = self.discreteToBox(otherAction)

    self.game.agent_left.setAction(otherAction)
    self.game.agent_right.setAction(action) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    if self.t >= self.t_limit:
      done = True

    if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
      done = True

    otherObs = None
    if self.multiagent:
      if self.from_pixels:
        otherObs = cv2.flip(obs, 1) # horizontal flip
      else:
        otherObs = self.game.agent_left.getObservation()

    info = {
      'ale.lives': self.game.agent_right.lives(),
      'ale.otherLives': self.game.agent_left.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_right.getObservation(),
      'otherState': self.game.agent_left.getObservation(),
      'otherAction': otherAction  ## the opponent action
    }

    if self.survival_bonus:
      return obs, reward+0.01, done, info
    return obs, reward, done, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()
    if self.training:          ##################################
      self.game.training = True ##################################"
    else:
      self.game.training = False
      
    if not self.upgrade:
      WORLD.upgrade = False
      WORLD.setup()

  def reset(self):
    self.init_game_state()
    return self.getObs()

  def checkViewer(self):
    # for opengl viewer
    if self.viewer is None:
      checkRendering()
      self.viewer = rendering.SimpleImageViewer(maxwidth=2160) # macbook pro resolution

  def render(self, mode='human', close=False):

    if PIXEL_MODE:
      if self.canvas is not None: # already rendered
        rgb_array = self.canvas
        self.canvas = None
        if mode == 'rgb_array' or mode == 'human':
          self.checkViewer()
          larger_canvas = upsize_image(rgb_array)
          self.viewer.imshow(larger_canvas)
          if (mode=='rgb_array'):
            return larger_canvas
          else:
            return

      self.canvas = self.game.display(self.canvas)
      # scale down to original res (looks better than rendering directly to lower res)
      self.canvas = downsize_image(self.canvas)

      if mode=='state':
        return np.copy(self.canvas)

      # upsampling w/ nearest interp method gives a retro "pixel" effect look
      larger_canvas = upsize_image(self.canvas)
      self.checkViewer()
      self.viewer.imshow(larger_canvas)
      if (mode=='rgb_array'):
        return larger_canvas

    else: # pyglet renderer
      if self.viewer is None:
        checkRendering()
        self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

      self.game.display(self.viewer)
      return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
    
  def get_action_meanings(self):
    return [self.atari_action_meaning[i] for i in self.atari_action_set]
    
  def upgrade_world(self):
  
    if self.upgrade:
      WORLD.upgrade_world()
    if not WORLD.upgrade:
      self.upgrade = False

class Slime3DPixelEnv(Slime3DEnv):
  from_pixels = True

class Slime3DAtariEnv(Slime3DEnv):
  from_pixels = True
  atari_mode = True

class Slime3DSurvivalAtariEnv(Slime3DEnv):
  from_pixels = True
  atari_mode = True
  survival_bonus = True

class SurvivalRewardEnv(gym.RewardWrapper):
  def __init__(self, env):
    """
    adds 0.01 to the reward for every timestep agent survives

    :param env: (Gym Environment) the environment
    """
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """
    adds that extra survival bonus for living a bit longer!

    :param reward: (float)
    """
    return reward + 0.01

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.

    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers

    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    return np.concatenate(list(self.frames), axis=2)

#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1
    render_mode = True
    if render_mode:
      env.render()
    sleep(0.01)

  return total_reward, t

def render_atari(obs):
  """
  Helper function that takes in a processed obs (84,84,4)
  Useful for visualizing what an Atari agent actually *sees*
  Outputs in Atari visual format (Top: resized to orig dimensions, buttom: 4 frames)
  """
  tempObs = []
  obs = np.copy(obs)
  for i in range(4):
    if i == 3:
      latest = np.copy(obs[:, :, i])
    if i > 0: # insert vertical lines
      obs[:, 0, i] = 141
    tempObs.append(obs[:, :, i])
  latest = np.expand_dims(latest, axis=2)
  latest = np.concatenate([latest*255.0] * 3, axis=2).astype(np.uint8)
  latest = cv2.resize(latest, (84 * 8, 84 * 4), interpolation=cv2.INTER_NEAREST)
  tempObs = np.concatenate(tempObs, axis=1)
  tempObs = np.expand_dims(tempObs, axis=2)
  tempObs = np.concatenate([tempObs*255.0] * 3, axis=2).astype(np.uint8)
  tempObs = cv2.resize(tempObs, (84 * 8, 84 * 2), interpolation=cv2.INTER_NEAREST)
  return np.concatenate([latest, tempObs], axis=0)

####################
# Reg envs for gym #
####################

register(
    id='SlimeVolley3D-v0',
    entry_point='slime3D_controller.slimevolley3D:Slime3DEnv'
)

register(
    id='SlimeVolley3DPixel-v0',
    entry_point='slime3D_controller.slimevolley3D:Slime3DPixelEnv'
)

register(
    id='SlimeVolley3DNoFrameskip-v0',
    entry_point='slime3D_controller.slimevolley3D:Slime3DAtariEnv'
)

register(
    id='SlimeVolley3DSurvivalNoFrameskip-v0',
    entry_point='slime3D_controller.slimevolley3D:SurvivalAtariEnv'
)

if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  left Agent:
  W - Jump
  A - Left
  D - Right

  right Agent:
  Up Arrow, Left Arrow, Right Arrow
  """

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  manualAction = [0, 0, 0] # forward, backward, jump
  otherManualAction = [0, 0, 0]
  manualMode = False
  otherManualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.UP:    manualAction[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

    if k == key.D:     otherManualAction[0] = 1
    if k == key.A:     otherManualAction[1] = 1
    if k == key.W:     otherManualAction[2] = 1
    if (k == key.D or k == key.A or k == key.W): otherManualMode = True

  def key_release(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.UP:    manualAction[2] = 0
    if k == key.D:     otherManualAction[0] = 0
    if k == key.A:     otherManualAction[1] = 0
    if k == key.W:     otherManualAction[2] = 0

  policy = BaselinePolicy() # defaults to use RNN Baseline for player

  env = Slime3DEnv()
  env.seed(np.random.randint(0, 10000))
  #env.seed(721)

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False

  while not done:

    if manualMode: # override with keyboard
      action = manualAction
    else:
      action = policy.predict(obs)

    if otherManualMode:
      otherAction = otherManualAction
      obs, reward, done, _ = env.step(action, otherAction)
    else:
      obs, reward, done, _ = env.step(action)

    if reward > 0 or reward < 0:
      print("reward", reward)
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.01)

    # make the game go slower for human players to be fair to humans.
    if (manualMode or otherManualMode):
      if PIXEL_MODE:
        sleep(0.01)
      else:
        sleep(0.02)

  env.close()
  print("cumulative score", total_reward)