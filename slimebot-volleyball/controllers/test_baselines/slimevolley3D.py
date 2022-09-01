"""
This is game is built on top of the slimevolleygym game. The code has been adjusted to 
make it work in a 3 coodrdinate setting and react with the simulator software Webots

Original game:
https://github.com/hardmaru/slimevolleygym

Simulator:
https://github.com/cyberbotics/webots
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
from controller import Supervisor ## Webots class to control nodes in the 3D scene area


np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5
MAXLIVES = 5 # game ends when one agent loses this many games

### Webots settings:
supervisor = Supervisor()
TIME_STEP = 32


class World:
    """ 
    The volleyball court setting.
    The setting is made according Webots coordinantes system (x: width, y: height, z: depth)
        
    All parameters are static except the depth that let the user the option of chosing an initial value in the
    interval [0, 24]
    
    :param update (bool): If "True" the agent will be incrementating the court depth during the training,
                          If "False" the agent will be training with the court maximum depth                          
    :param width: The width (x-axis) of the court
    :param height: The height (y-axis) of the court
    :param depth: The depth (z-axis) of the court.
    :param max_deepth: The maximum depth of the court defautly 24
    :param step: Defines how much depth should be incremented if the update_world function is called
    :param wall_depth: The court fence width
    :param wall_height: The court fence height
    :param wall_depth: The court fence depth (dynamic as the depth)
    :param player_v (x/y/z): The palyer speed according the specific axis
    :param max_ball_v: The maximum speed the ball can take
    :param gravity: The world gravity        
    """

    def __init__(self, update = True):
     
        self.width = 48
        self.height = self.width
        self.max_depth = self.width/2
        self.step = -np.inf
        self.depth = -np.inf  
        self.wall_width = 1.0 
        self.wall_height = 2.0
        self.wall_depth = -np.inf
        self.player_vx = 17.5
        self.player_vy = 13.5
        self.player_vz = 17.5
        self.max_ball_v = 15*1.5       
        self.gravity = -9.8*2*1.5   
        self.update = update
        self.setup()
  
    def setup(self, n_update = 4, init_depth = 6):
        """
        Function that set up the depth of the environement before and during the training.
        If update = True the depth is setup, else the depth is set equal to the maximum depth
        
        :param n_upgr: The number of time the depth will be updated during the training
        :param init_depth: The intial depth of the court at the beginning of the training
        """
        
        if not self.update:
            self.wall_depth = self.depth = self.max_depth      
        else: 
            self.step = self.max_depth/n_update  
            self.wall_depth  = self.depth = init_depth  

    
    def update_world(self):
        """
        Function that update the depth of the court if the parameter "update" is True
        """
        if self.update:         
            self.depth +=  self.step
            self.wall_depth = self.depth      
          
        if self.depth >= self.max_depth:
            self.depth = self.max_depth
            self.wall_depth = self.depth                              
            self.update = False
          

WORLD = World() # 
      
class DelayScreen:
    """ 
    initially the ball is held still for INIT_DELAY_FRAMES(30) frames
    """  
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
    """ 
    used for the ball, and also for the curved stub above the fence 
    :params x,y,z: represent the current position of the particule
    :params prev_x, prev_y, prev_z: represent the positon of the particule at the previous timestep
    :params vx, vy, vz: represent the speed coordinates of the particule
    :param r: represents the radius of the particule following x_axis
    :param name: The name of the particule, allows webots to recognize which object the particule represents
                 in the 3D scene.
    :param particule: Enables Webots supervisor method to control the particule in the 3D view
    :param location: Controls the particule location in Webots 3D scene
    """
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
         
        # Webots settings         
        self.particule = supervisor.getFromDef(name)
        self.location = self.particule.getField("translation")
        self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    def move(self):
        """
        Control the movement of the particule
        """
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP  
        self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    def applyAcceleration(self, ax, ay, az):
        """
        Apply acceleration to the particule when a collision occurs.
        :params ax, ay, az: the coordinates of the acceleration vector
        """
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP
        self.vz += az * TIMESTEP * (WORLD.depth/WORLD.max_depth) #Keep the z-axis proportional to the actual depth    
    
    def checkEdges(self):
        """
        Check that the partcicule respect the game rule
        """
        
        # Avoid the particule x-location to go beyond the court width
        if (self.x <= (self.r-WORLD.width/2)):
            self.vx *= -FRICTION
            self.x = self.r-WORLD.width/2+NUDGE*TIMESTEP    
        if (self.x >= (WORLD.width/2-self.r)):
            self.vx *= -FRICTION;
            self.x = WORLD.width/2-self.r-NUDGE*TIMESTEP
        
        # Make sure the particule moves properly on the z-axis according to the current depth  
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
    
        #If the particule hit the floor, prevent it from crossing the floor
        if (self.y<=(self.r)):
            self.vy *= -FRICTION
            self.y = self.r+NUDGE*TIMESTEP         
            self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
          
            if (self.x <= 0):
                return -1 # The left player loses a life
            else:
                return 1 # The right player loses a life
                
        # Avoid the particule to go beyond the world maximum height    
        if (self.y >= (WORLD.height-self.r)):
            self.vy *= -FRICTION
            self.y = WORLD.height-self.r-NUDGE*TIMESTEP
          
        # Avoid the particule to cross the fence:
        if ((self.x <= (WORLD.wall_width/2+self.r)) and (self.prev_x > (WORLD.wall_width/2+self.r)) and (self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = WORLD.wall_width/2+self.r+NUDGE*TIMESTEP
    
        if ((self.x >= (-WORLD.wall_width/2-self.r)) and (self.prev_x < (-WORLD.wall_width/2-self.r)) and (self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = -WORLD.wall_width/2-self.r-NUDGE*TIMESTEP
          
        self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
               
        return 0;
    
    def getDist2(self, p):
        """
        Compute de squared distance with another object
        """
        dz = p.z - self.z
        dy = p.y - self.y
        dx = p.x - self.x    
        return (dx*dx+dy*dy+dz*dz)
    
    def isColliding(self, p): 
        """
        Returns true if the particule is colliding with a given object
        """
        r = self.r+p.r
        if WORLD.depth != 0:
            # if distance is less than total radius and the depth, then colliding.
            return (r*r > self.getDist2(p) and (self.z*self.z <= WORLD.wall_depth * WORLD.wall_depth)) 
        else:
            return (r*r > self.getDist2(p))
      
    def bounce(self, p):
        """
        bounce two particules that have collided (this and that)
        :param p: objects with which particule has collided
        """
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
        """
        Check the limitspeed of the particule
        """
        mag2 = self.vx*self.vx+self.vy*self.vy+self.vz*self.vz;
        if (mag2 > (maxSpeed*maxSpeed) ):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vz /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
            self.vz *= maxSpeed * (WORLD.depth/WORLD.max_depth) # Make qure the z-coordinate stays proportional to the court depth    
        if (mag2 < (minSpeed*minSpeed) ):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vz /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed
            self.vz *= minSpeed * (WORLD.depth/WORLD.max_depth) # Make qure the z-coordinate stays proportional to the court depth


class RelativeState:
    """
    keeps track of the mobile objects in the game.
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
    """ 
    keeps track of the agent in the game. note this is not the policy network 
    
    :params x, y, z: locations coordinates
    :params vx, vy, vz: speed coordinates
    :param desired_v (x/y/z):
    :param state:Instance of the Relative class used to track mobile objects in the scene
    :param life: Number of life during the game (maximum 5)
    :param name: The name of the agent, allows webots to recognize which object in the 3D scene is represented by the agent
    :param agent: Allows to track the agent in Webots
    :param location: controls the location of the agent in Webots 3D scene
    """
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
        
        # Webots settings
        self.agent = supervisor.getFromDef(name)
        self.location = self.agent.getField("translation")
        self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
        
    def lives(self):
        """
        return the number of avialabe lives
        """
        return self.life
    
    def setAction(self, action):
        """
        set the possible movements according to the given action 
        """
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
        """
        Control the movement of the agent
        """
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
        """
        
        """
        self.vy += GRAVITY * TIMESTEP   
        if (self.y <=  NUDGE*TIMESTEP):
            self.vy = self.desired_vy
    
        self.vx = self.desired_vx*self.dir
        self.vz = self.desired_vz   
        self.move()
    
        #Make sure the agent doesn't cross the floor
        if (self.y <= 0):
            self.y = 0;
            self.vy = 0;
    
        # stay in their own half and the court width:
        if (self.x*self.dir <= (WORLD.wall_width/2+self.r) ):
            self.vx = 0;
            self.x = self.dir*(WORLD.wall_width/2+self.r)
    
        if (self.x*self.dir >= (WORLD.width/2-self.r) ):
            self.vx = 0;
            self.x = self.dir*(WORLD.width/2-self.r)
          
        # stay in the court depth area:        
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
        """ 
        Normalized to side, appears different for each agent's perspective
        """
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
    """
    Except that an agent is given as baseline, the random agent will serve as the baseline
    """
    def __init__(self, agent = None, env = None):
        self.agent = agent
        self.env = env
    
    def predict(self, obs):
        if self.agent is None:
            return self.env.action_space.sample()
        else:
            action, _ = self.agent.predict(obs)
        return action

class Game:
    """
    The game setting.
    
    :param np_random: Endling the randomazition of the domain
    :param training (bool): If "True" the ball will always be launched on the side of the learning agent (the yellow)
    :param ball: An instance of the Particule class that will serve as the game ball
    :param fenceStub:An instance of the Particule class that will serve as the curve shape on the top of the fence
    :agent (left/right) : Agents of the game (left: blue, right: yellow)
    :delayScreen: make sure the ball stay still for a period before getting launched
    """
    def __init__(self, np_random=np.random, training = False):
        self.ball = None
        self.fenceStub = None
        self.agent_left = None
        self.agent_right = None
        self.delayScreen = None
        self.np_random = np_random       
        self.training = training
        self.reset()
    
    def reset(self):
        """
        Brining the game to the initial set up
        
        NOTE: Names (strings parameters) given to objects are set in a way that Webots will recognize them. Modifying 
        them will require to rectify in Webots too (which could be tadeous), otherwise errors will occur.
        """      
        self.fenceStub = Particle(0, WORLD.wall_height, 0, 0, 0, 0, WORLD.wall_width/2, "FENCESTUB");
        if self.training: # Launch the ball on the trainer's side
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
        """reinitialze mobile objects positions at the beginning of a new match
        """
        if self.training:
            ball_vx = self.np_random.uniform(low=0, high=20)
        else:
            ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        ball_vz = self.np_random.uniform(low=-10, high=10) * (WORLD.depth/WORLD.max_depth)
        self.ball = Particle(0, (WORLD.width/4)-1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
        self.delayScreen.reset()
    
    def step(self):
        """ 
        Game main game loop 
        """    
        self.betweenGameControl()
        self.agent_left.update()
        self.agent_right.update()
    
        if self.delayScreen.status():
            self.ball.applyAcceleration(0, WORLD.gravity, 0)
            self.ball.limitSpeed(0, WORLD.max_ball_v)
            self.ball.move()
        #Check the collision of the ball with agents and the fenceStub
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
                self.agent_right.life -= 1 # yellow agent lose a life
            else:
                self.agent_left.emotion = "sad"
                self.agent_right.emotion = "happy"
                self.agent_left.life -= 1 #blue agent lose a life
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
    The main game environment. This enviroment works, with some additional differences, the same
    as the original version (https://github.com/hardmaru/slimevolleygym)

    
    The function "update_world", if called while training, allows the trainer (the right agent)
    to increment the environment depth.
    
    The game ends when an agent loses 5 lives (or at t=3000 timesteps). If running in Webots,
    there are lifes counter for both agents on the top corners of the 3D view.
    """
    
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
    survival_bonus = False # Depreciated: augment reward, easier to train
    multiagent = True # optional args anyways
    
    """
    :param training (bool): If True the ball will always be launched on the side of the trainer,
                            allow to boost the training speed
    :param update (bool): If "True" the agent will start its training with a selected initial depth
                          and will be increasing when the function "update_world" is called. If
                          If "False" the agent start the training directelyvwith the world maximal
                          depth.
    :param t (int) : keeps track of the cureent timestep during the game
    :pram t_limit (int): Maximum number of timesteps for a game
    :world (World class instance): Keeps track of the physical aspect of the court
    :ret (int): cumulative reward at the current timestep                                             
    """

    def __init__(self, training = False, update = False):    
        
        
        self.t = 0
        self.t_limit = 3000
        self.atari_mode = False
        self.num_envs = 1
        self.training = training
        self.update = update
        self.world = WORLD #
        self.ret = 0
           
        
        if self.atari_mode: ## Not sure there is an atari compatibility with the 3d version
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
        
         
        self.game = Game(training = self.training)
        self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    
        self.policy = BaselinePolicy(env = self) # the “bad guy”   
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
        # convert discrete action n into the actual quintupled action
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
        
        #drawing agent names and lives on the top corners of Webots 3D views
        self.draw_agent_name()
        self.draw_lives(self.game.agent_left.life, self.game.agent_right.life)
        self.draw_time(int((self.t_limit-self.t)/100))
             
        done = False
        self.t += 1
        supervisor.step(TIME_STEP) ## Important for Webots world similations
        
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
        self.ret += reward
    
        obs = self.getObs()
        
        epinfos = {} ## important for te stablebaselines PPO2
        if self.t >= self.t_limit:
            done = True   
            epinfos = {'r': self.ret, 'l': self.t }
              
        if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
            done = True
            epinfos = {'r': self.ret, 'l': self.t }
            
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
        info.update(epinfos)
                  
        if self.survival_bonus:
              return obs, reward+0.01, done, info
        return obs, reward, done, info

    def init_game_state(self):
        self.t = 0
        self.ret = 0
        self.game.reset()
        if self.training:          ##################################
            self.game.training = True ##################################"
        else:
            self.game.training = False
          
        if not self.update:
            self.world.update = False
            self.world.setup()

    def reset(self):
        self.draw_time(int(self.t_limit/100))
        self.init_game_state()
        return self.getObs()
   
    def get_action_meanings(self):
        return [self.atari_action_meaning[i] for i in self.atari_action_set]
    
    def update_world(self):  
        if self.update:
            self.world.update_world()
        if not self.world.update:
            self.update = False
            self.training = False
      
    def draw_agent_name(self):
        """
        Display the names of the agents (Blue on the left and Yellow on the right) in Webots 3d
        view
        """
        supervisor.setLabel(
                            0, # LabelId
                            "Yellow", # Text to display
                            0.76 - (len("Yellow") * 0.01),  # X position
                            0.01,  # Y position
                            0.1,  # Size
                            0xFFFF00,  # Color
                            0.0,  # Transparency
                            "Tahoma",  # Font
                            )
        supervisor.setLabel(
                            1, # LabelId
                            "Blue", # Text to display
                            0.05,  # X position
                            0.01,  # Y position
                            0.1,  # Size
                            0x0000FF,  # Color
                            0.0,  # Transparency
                            "Tahoma",  # Font
                            )        
    

    def draw_lives(self, left_life: int, right_life: int):
        """
        Display agents' remaining lives in Webots 3D view
        """  
        supervisor.setLabel(
                            2, # LabelId
                            "remaining lives: " + str(right_life-1), # Text to display
                            0.7,  # X position
                            0.05,  # Y position
                            0.1,  # Size
                            0xFFFF00,  # Color
                            0.0,  # Transparency
                            "Tahoma",  # Font
                            ) 
        supervisor.setLabel(
                            3, #LabelId
                            "remaining lives: " + str(left_life-1), # Text to display
                            0.05,  # X position
                            0.05,  # Y position
                            0.1,  # Size
                            0x0000FF,  # Color            
                            0.0,  # Transparency
                            "Tahoma",  # Font
                            )
                      
    def draw_time(self, time: int):
        """
        Display the current match time (a countdown of 30 secondes)
        """
        supervisor.setLabel(
                            4, # Label
                            "00:"+ str(time), # Text to display
                            0.45, # X position
                            0.01, # y position
                            0.1, # size
                            0x000000, # Color
                            0.0, # transparency
                            "Arial", # Font
                            )

    def draw_event_messages(self, messages):
        """
        Display the event messages from queue       
        """
        if messages:
            supervisor.setLabel(
                               5, # LabelId
                              "New match!", # Text to display
                              0.01, # X position
                              0.95 - ((len(messages) - 1) * 0.025), # Y position
                              0.05, # size
                              0xFFFFFF, # Color
                              0.0, # Transparency
                              "Tahoma", # Font
                              )          