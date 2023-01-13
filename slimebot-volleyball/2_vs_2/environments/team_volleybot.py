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
import copy
try:
    from controller import Supervisor ## Webots method to control nodes in the 3D scene area
    WEBOTS_MODE = True
except:
    WEBOTS_MODE = False


np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5
MAXLIVES = 5 # game ends when one agent loses this many games

### Webots settings:
if WEBOTS_MODE:
    supervisor = Supervisor()
TIME_STEP = 32


class World:
    """ 
    The volleyball court setting.
    The setting is made according Webots coordinantes system (x: width, y: height, z: depth)
        
    All parameters are static except the depth that let the user the option of chosing an initial value in the
    interval [0, 24]
    
    :param upgrade (bool): If "True" the agent will be upgrading the court depth during the training,
                           If "False" the agent will be training with the court maximum depth
    :param width: The width (x-axis) of the court
    :param height: The height (y-axis) of the court
    :param depth: The depth that the agent will be currently training value between 0 and max_depth
    :param max_deepth: The maximum depth of the court
    :param step: Defines how much depth should increase if the update function is called
    :param wall_depth: The court fence width
    :param wall_height: The court fence height
    :param wall_depth: The court fence depth (dynamic as the depth)
    :param player_v (x/y/z): The palyer speed according the specific axis
    :param max_ball_v: The maximum speed the ball can take
    :param gravity: The world gravity speed        
    """

    def __init__(self, update = True):
     
        self.width = 24*2
        self.height = self.width
        self.max_depth = self.width/2
        self.step = -np.inf
        self.depth = -np.inf  
        self.wall_width = 1.0 
        self.wall_height = 2.0
        self.wall_depth = -np.inf
        self.player_vx = 10*1.75
        self.player_vy = 10*1.35
        self.player_vz = 10*1.75
        self.max_ball_v = 15*1.5       
        self.gravity = -9.8*2*1.5   
        self.update = update
        self.stuck = False
        self.setup()
  
    def setup(self, n_update = 4, init_depth = 6):
        """
        Function that set up the depth of the environement before and during the training.
        if upgrade = True the depth is setup, else the depth is set equal to the maximum depth
        
        :param n_upgr: The number of time the depth will be updated during the training
        :param init_depth: The intial depth of the court at the beginning of the training
        """
        
        if not self.update:
            self.wall_depth = self.depth = self.max_depth     
        elif self.stuck:
            self.step = 0
            self.wall_depth = self.depth = init_depth 
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
          
WORLD = World()   
      
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
    used for the ball, and also for the round stub above the fence 
    :params x,y,z: represent the current position of the particule
    :params prev_x, prev_y, prev_z: represent the positon of the particule at the previous timestep
    :params vx, vy, vz: represent the speed coordinates of the particule
    :param r: represents the radius of the particule
    :param name: The name of the particule, allows webots to recognize which object in the 3D scene the particule represents
    :param particule: Enables Webots supervisor method to control the particule in the 3D view
    :param location: Controls the particule location in Webots 3D window
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
        if WEBOTS_MODE:         
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
        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    def applyAcceleration(self, ax, ay, az):
        """
        Apply acceleration to the particule when a collision occur.
        :params ax, ay, az: the coordinates of the acceleration vector
        """
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP
        self.vz += az * TIMESTEP * (WORLD.depth/WORLD.max_depth) #Keep the z-axis proportional to the actual depth    
    
    def checkEdges(self):
        """
        Check that the partcicule respect the game rule
        """
        
        #If the particule x-location goes beyond the court width lock it at the limit
        if (self.x <= (self.r-WORLD.width/2)):
            self.vx *= -FRICTION
            self.x = self.r-WORLD.width/2+NUDGE*TIMESTEP    
        if (self.x >= (WORLD.width/2-self.r)):
            self.vx *= -FRICTION;
            self.x = WORLD.width/2-self.r-NUDGE*TIMESTEP
        
        # Make sure the particule moves correctly on the z-axis according to the current court depth  
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
    
        #If the particule hit the floor, prevent it from crossing the ground
        if (self.y<=(self.r)):
            self.vy *= -FRICTION
            self.y = self.r+NUDGE*TIMESTEP 
            if WEBOTS_MODE:        
                self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
          
            if (self.x <= 0):
                return -1 # The left player loses a life
            else:
                return 1 # The right player loses a life
                
        #Avoid the particule to go beyond the world height    
        if (self.y >= (WORLD.height-self.r)):
            self.vy *= -FRICTION
            self.y = WORLD.height-self.r-NUDGE*TIMESTEP
          
        # avoid the particule from crosssing the fence:
        if ((self.x <= (WORLD.wall_width/2+self.r)) and (self.prev_x > (WORLD.wall_width/2+self.r)) and (self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = WORLD.wall_width/2+self.r+NUDGE*TIMESTEP
    
        if ((self.x >= (-WORLD.wall_width/2-self.r)) and (self.prev_x < (-WORLD.wall_width/2-self.r)) and (self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = -WORLD.wall_width/2-self.r-NUDGE*TIMESTEP
            
        if WEBOTS_MODE:  
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
          #print((self.isColliding(p)))  
          self.x += abx
          self.y += aby
          self.z += abz    
          if WEBOTS_MODE:      
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



        
class Team():

    def __init__(self, dir, name, n_mates = 2):
        self.name = name
        self.n_mates = n_mates
        self.team  = {}  
        self.dir = dir
        self.setTeam()
        self.life = MAXLIVES     
        
    def lives(self):
    
        return self.life       
        
    def setTeam(self):
        for i in range(self.n_mates):
            self.team[self.name+str(i+1)] = Agent(self.dir, (self.dir*9) + (self.dir*i*6), 0, 
                                                 0, 
                                                 self.name+str(i+1)
                                                 )
            """self.team[self.name+str(i+1)] = Agent(self.dir, (self.dir*12), 0, 
                                                 self.dir*4 - self.dir * i*8, 
                                                 self.name+str(i+1)
                                                 )"""
                                                 
        for agt in self.team:
            self.team[agt].getMates(self.team)
                                                 
                                              
    def setAction(self, actions: list):
       
        for i in range(len(actions)):
            
            agent = self.team[self.name+str(i+1)]
            agent.setAction(list(actions[i]))
             
    def teamMove(self):
        for i in range(self.n_mates):
            agent = self.team[self.name+str(i+1)]
            agent.move()
             
             
    def update(self):
         for i in range(self.n_mates):
             agent = self.team[self.name+str(i+1)]
             agent.update()
            
    def getObservations(self, ball, opponents):
        obs = []
        for i in range(self.n_mates):
            agent = self.team[self.name+str(i+1)]             
            obs.append(agent.getObs(ball, opponents))             
        return [ob.reshape(1,-1) for ob in obs]
        #return obs
                 
    def getTeamState(self):
        states = []
        for i in range(self.n_mates):
            agent = self.team[self.name+str(i+1)]
            states += [agent.x, agent.y, agent.z, agent.vx, agent.vy, agent.vz]
              
        return states
          
    """def updateTeamState(self, ball, opponents):
         
         for i in range(self.n_teamates):
             agent = self.team[self.name+str(i+1)]
             agent.updateState(ball, opponents)"""
          
            

class Agent():
    """ 
    keeps track of the agent in the game. note this is not the policy network 
    :params x, y, z: locations coordinates
    :params vx, vy, vz: speed coordinates
    :param desired_v (x/y/z):
    :param state:
    :param life: Number of life during the game (maximum 5)
    :param name: The name of the agent, allows webots to recognize which object in the 3D scene is represented by the agent
    :param agent: Allows Webots to track the agent position in the 3D window
    :param location: controlls the location of the agent in Webots 3D view
    """
    def __init__(self, dir, x, y, z, name):
    
        self.dir = dir # -1 means left, 1 means right player for symmetry.
        self.x = x
        self.y = y
        self.z = z
        
        """self.prev_x = None
        self.prev_y = None
        self.prev_z = None"""
        
        self.r = 1.5
        self.name = name
        self.side = int(name[-1])
        self.bonus = 0
        self.malus = 0
        
        self.BallCollisionS = 0
        
        
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.desired_vz = 0
        #self.state = RelativeState()
        self.emotion = "happy"; # hehe...
        self.life = MAXLIVES
        self.mates = {}
        if WEBOTS_MODE: 
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
        action = list(action)
        #print(action)
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
        #self.prev_x, self.prev_y, self.prev_z = self.x, self.y, self.z
        
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP 
        if WEBOTS_MODE:      
            self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    def step(self):
        #self.prev_x, self.prev_y, self.prev_z = self.x, self.y, self.z
        
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP
        if WEBOTS_MODE:
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
    
        #Make sure the agant is not crossing the floor
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
        if WEBOTS_MODE:     
            self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
    
    
        
        
    def getObservation(self):
        return self.state.getObservation()
        
    def getMates(self, team):
        mates = team.copy()
        del mates[self.name]
        for i in mates:
            self.mates[i] = mates[i]
            
    
    def getObs(self, ball, opponents):
        obs = [self.x*self.dir, self.y, self.z, self.vx*self.dir, self.vy, self.vz]      
        for i in self.mates:
            obs += [self.mates[i].x*self.dir, self.mates[i].y, self.mates[i].z, 
                    self.mates[i].vx*self.dir,  self.mates[i].vy, self.mates[i].vz]
                    
        obs += [ball.x *self.dir, ball.y, ball.z, ball.vx*self.dir, ball.vy, ball.vz]
        
        op = opponents.getTeamState()
        for i in range(0, len(op), 3):
            op[i] *= -self.dir
        obs += op
            
        return np.array(obs)
        
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
            return (r*r >= self.getDist2(p)) and (self.z*self.z <= WORLD.wall_depth * WORLD.wall_depth) 
        else:
            return (r*r >= self.getDist2(p))
      
    def collision(self, p):
        """
        collision with another mates two particules that have collided (this and that)
        :param p: objects with which particule has collided
        """
        abx = self.x-p.x
        aby = self.y-p.y
        abz = self.z-p.z
        abd = math.sqrt(abx*abx+aby*aby+abz*abz)
        if abd != 0:
            
            abx /= abd # normalize
            aby /= abd
            abz /= abd
            nx = abx # reuse calculation
            ny = aby
            nz = abz
            abx *= NUDGE*0.4
            aby *= NUDGE*0.4
            abz *= NUDGE*0.4
         
        else:
             abx = aby = abz = 1.7
        
        
        
        while(self.isColliding(p)):
        
          
          self.x += abx
          self.y += aby
          self.z += abz          
          if WEBOTS_MODE:
              self.location.setSFVec3f([self.x*0.1, self.y*0.1, self.z*0.1])
          
          p.x -= abx
          p.y -= aby
          p.z -= abz     
          if WEBOTS_MODE:     
              p.location.setSFVec3f([p.x*0.1, p.y*0.1, p.z*0.1])
          
    """def selfDist(self):
    
        dz = self.prev_z - self.z
        dy = self.prev_y - self.y
        dx = self.prev_x - self.x    
        
        return (dx*dx+dy*dy+dz*dz)"""
    def checkSide(self):
        
        return (self.z // 12) + 1  == (self.side -1)
        
        
                   
          
       
        

class BaselinePolicy:
    """
    Except that a agent is given as baseline, the random agent will serve as the baseline
    """
    def __init__(self, agent = None, env = None):
        self.agent = agent
        self.env = env
    
    def predict(self, obs):
        if self.agent is None:
            actions = []
            for i in range(2):
                actions.append(self.env.action_space.sample())
        else:
            actions, _ = self.agent.predict(obs)
        return actions

class Game:
    """
    The game setting.
    :param np_random: Endling the randomazition of the domain
    :param training (bool): if set "True" the ball will always be launched on the side of the learning agent (the yellow)
    :param ball: An instance of the Particule class that will serve as the game ball
    :param fenceStub:An instance of the Particule class that will serve as the curve shape on the top of the fence
    :agent (left/right) : Agents of the game (left: blue, right: yellow)
    :delayScreen: make sure the ball stay still for a period before getting launched
    """
    def __init__(self, np_random=np.random, training = False, eval_mode = False ):
        self.ball = None
        self.fenceStub = None
        self.team_left = None
        self.team_right = None
        self.delayScreen = None
        self.np_random = np_random       
        self.training = training
        self.eval_mode = eval_mode
        self.reset()
        self.BallCollision = [0,0]
        self.NotSide = [0,0]
        self.match = 0
        
    
    def reset(self):
        """
        Brining the game to the initial set up
        
        NOTE: Names (strings) given to objects are set such that Webots will recognize them. Modifying them will require to
              rectify in Webots too (which could be tadeous), otherwise errors will occur.
        """
        self.match = 0      
        self.fenceStub = Particle(0, WORLD.wall_height, 0, 0, 0, 0, WORLD.wall_width/2, "FENCESTUB");
        if self.training:
          ball_vx = self.np_random.uniform(low=0, high=20)
        else:
          ball_vx = self.np_random.uniform(low=-20, high=20)       
        ball_vy = self.np_random.uniform(low=10, high=25)
        ball_vz = self.np_random.uniform(low=0, high=10) * (WORLD.depth/WORLD.max_depth)
        #ball_vz = self.np_random.uniform(low=-10, high=10) * (WORLD.depth/WORLD.max_depth)
        self.ball = Particle(0, (WORLD.width/4)-1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
        self.team_left = Team(-1, "BLUE")
        self.team_right = Team(1, "YELLOW")
        
        #self.team_left.updateState(self.ball, self.agent_right)
        #self.team_right.updateState(self.ball, self.agent_left)   
        self.delayScreen = DelayScreen()
    
    def newMatch(self):
        """reinitialze main objects positions at the beginning of a new match
        """
        self.match += 1
        
        if self.training:
            ball_vx = self.np_random.uniform(low=0, high=20)
        else:
            ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        if self.match % 2 == 0:
            ball_vz = self.np_random.uniform(low=0, high=10) * (WORLD.depth/WORLD.max_depth)
        else:
            ball_vz = self.np_random.uniform(low=-10, high=0) * (WORLD.depth/WORLD.max_depth)
        self.ball = Particle(0, (WORLD.width/4)-1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
        self.delayScreen.reset()
    
    def step(self):
        """ 
        Game main game loop 
        """    
        self.betweenGameControl()
        self.team_left.update()
        self.team_right.update()
        self.BallCollision = [0,0]
        self.NotSide = [0,0]
    
        if self.delayScreen.status():
            self.ball.applyAcceleration(0, WORLD.gravity, 0)
            self.ball.limitSpeed(0, WORLD.max_ball_v)
            self.ball.move()
        #Check the collision of the ball with agents and the fenceStub
        for agent in self.team_left.team:
            if (self.ball.isColliding(self.team_left.team[agent])):
                self.ball.bounce(self.team_left.team[agent])
            mates = self.team_left.team[agent].mates
            for i in mates:
                if (self.team_left.team[agent].isColliding(mates[i])):
                    self.team_left.team[agent].collision(mates[i])
                     
                
        for agent in self.team_right.team:
            
            mates = self.team_right.team[agent].mates
            if (self.ball.isColliding(self.team_right.team[agent])):
                self.ball.bounce(self.team_right.team[agent])
                
                self.BallCollision[int(agent[-1]) - 1 ] = 1
            
            #print(self.team_right.team[agent].checkSide())  
            #print(self.team_right.team[agent].name)     
            
            if (self.team_right.team[agent].checkSide()):
                self.team_right.team[agent].bonus = 0.001
                
                self.NotSide[int(agent[-1]) - 1 ] = 0
            else:
                self.team_right.team[agent].malus = -0.01   
                self.NotSide[int(agent[-1]) - 1 ] = 1   
                  
            
            for i in mates:
                              
                if (self.team_right.team[agent].isColliding(mates[i])):
                    self.team_right.team[agent].collision(mates[i])
                    if (mates[i].checkSide()):
                        self.team_right.team[agent].malus = -0.05
                    elif (self.team_right.team[agent].checkSide()):
                        mates[i].malus = -0.05                  
                    
                    #print("collision")                                                                          
        
        self.fenceStub.z = self.ball.z
        if (self.ball.isColliding(self.fenceStub)):         
            self.ball.bounce(self.fenceStub)
    
        # negated, since we want reward to be from the persepctive of right agent being trained.
        result = -self.ball.checkEdges()
        
        if result == -1:
        
            if self.ball.z <= 0:
                agent_results =   [-2, 0]
            else:
                agent_results = [0, -2]
        else:
            agent_results = [result for i in range(2)]
        
        Bonus = []
        for agent in self.team_right.team:
            Bonus.append(self.team_right.team[agent].bonus +self.team_right.team[agent].malus)
            
        for agent in self.team_right.team:
            self.team_right.team[agent].malus = 0
            self.team_right.team[agent].bonus = 0
            
        
        
        if (result != 0 ):
            self.newMatch() # not reset, but after a point is scored
            if result < 0: # baseline agent won
                self.team_left.emotion = "happy"
                self.team_right.emotion = "sad"
                self.team_right.life -= 1 # yellow agent lose a life
            else:
                self.team_left.emotion = "sad"
                self.team_right.emotion = "happy"
                self.team_left.life -= 1 #blue agent lose a life
                
            if self.eval_mode:
                return [result for i in range(2)]
            
            #print(agent_results)   
            return [agent_results[i] + Bonus [i] for i in range(len(Bonus)) ]
    
        # update internal states (the last thing to do)
        #self.agent_left.updateState(self.ball, self.agent_right)
        #self.agent_right.updateState(self.ball, self.agent_left)
        #print([result+ i for i in Bonus])
        if self.eval_mode:
            return [result for i in range(2)]
            
        return [result+ i for i in Bonus]
        
  
    def betweenGameControl(self):
        agent = [self.team_left, self.team_right]
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
          
      
      

class TeamVolleyBot(gym.Env):
    """
    The main game environment. This enviroment operate the same way as the original slimevolley
    gym environnment with just a few differences.
    
    The function "update_world", if called while training, allows the trainer (the right agent)
    to modify the environment depth.
    
    The game ends when an agent loses 5 lives (or at t=3000 timesteps). If running in Webots,
    there is a life counter on the corners of the 3D view.
    """
    metadata = {
      'render.modes': ['human', 'rgb_array', 'state', 'webots'],
      'video.frames_per_second' : 50
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
    survival_bonus = False # Depreciated: augment reward, easier to train
    multiagent = True # optional args anyways
    
    """
    :param training (bool): If True the ball will always be launched on the side of the trainer,
                            allow to boost the training speed
    :param update (bool): If "True" the agent will start its training with a selected initial depth
                          and will be increasing when the function "update_world" is called. If
                          If "False" the agent start the training directelyvwith the world maximal
                          depth.                          
    """

    def __init__(self, training = False, update = False, n_agents = 2, eval_mode = False):
   
        """
        Reward modes:
    
        net score = right agent wins minus left agent wins
    
        0: returns net score (basic reward)
        1: returns 0.01 x number of timesteps (max 3000) (survival reward)
        2: sum of basic reward and survival reward
    
        0 is suitable for evaluation, while 1 and 2 may be good for training
    
        Setting multiagent to True puts in info (4th thing returned in stop)
        the otherObs, the observation for the other agent. See multiagent.py
        """
        
        self.t = 0
        self.t_limit = 3000
        self.atari_mode = False
        self.num_envs = 1
        self.training = training
        self.update = update
        self.world = WORLD #
        self.n_agents = n_agents
        self.ret = [0,0]
        self.BallCollision = [0,0]
        self.NotSide = [0,0]
        self.eval_mode = eval_mode
        
        if self.atari_mode: ## Not sure there is an atari compatibility with the 3d version
            action_space = spaces.Discrete(3)
        else:
            action_space = spaces.MultiBinary(5)
        
        self.action_space = action_space #[action_space for _ in range(self.n_agents)]
    
        if self.from_pixels:
            setPixelObsMode()
            observation_space = spaces.Box(low=0, high=255,
            shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        else:
            high = np.array([np.finfo(np.float32).max] * 30)
            observation_space = spaces.Box(-high, high)
            
        self.observation_space = observation_space #[observation_space for _ in range(self.n_agents)]
        
        self.previous_rgbarray = None   
        self.game = Game(training = self.training)
        self.ale = self.game.team_right # for compatibility for some models that need the self.ale.lives() function
    
        self.policy = BaselinePolicy() # the “bad guy”   
        self.viewer = None
    
        # another avenue to override the built-in AI's action, going past many env wraps:
        self.otherActions = None
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game = Game(np_random=self.np_random, training = self.training)
        self.ale = self.game.team_right # for compatibility for some models that need the self.ale.lives() function
        return [seed]

    def getObs(self):
        if self.from_pixels:
            obs = self.render(mode='state')
            self.canvas = obs
        else:
            obs = self.game.team_right.getObservations(self.game.ball, self.game.team_left)
        return obs

    def discreteToBox(self, n):
        # convert discrete action n into the actual quintupled action
        actions = []
        for act in n:
            if isinstance(act, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
                if len(act) == 5:
                    actions.append(act)
        if len(actions) == len(n):
            return n
        
        for act in  n:    
            assert (int(act) == act) and (act >= 0) and (act < 18)
            
        return [self.action_table[act] for act in n]

    def step(self, actions, otherActions=None):
        """
        baseAction is only used if multiagent mode is True
        note: although the action space is multi-binary, float vectors
        are fine (refer to setAction() to see how they get interpreted)
        """
        
        #drawing agent names and lives on the top corners of Webots 3D views
        if WEBOTS_MODE:
            self.draw_team_name()
            self.draw_lives(self.game.team_left.life, self.game.team_right.life)
            self.draw_time(int((self.t_limit-self.t)/100))
            supervisor.step(TIME_STEP) ## Important for Webots world similations
             
        done = False
        self.t += 1
        
        
        if self.otherActions is not None:
            otherActions = self.otherActions          
        if otherActions is None: # override baseline policy
            observations = self.game.team_left.getObservations(self.game.ball, self.game.team_right)
            
            for i in range(2):
                otherActions = self.policy.predict(observations)
        if self.atari_mode:
            actions = self.discreteToBox(actions)
            otherActions = self.discreteToBox(otherActions)
    
        self.game.team_left.setAction(otherActions)
        self.game.team_right.setAction(actions) # external agent is agent_right
    
        reward = self.game.step()
        self.ret = [reward[i] + self.ret[i] for i in range(len(reward))]
        self.BallCollision = [self.game.BallCollision[i] + self.BallCollision[i] for i in range(2)]
        self.NotSide = [self.game.NotSide[i] + self.NotSide[i] for i in range(2)]
        
    
        obs = self.getObs()
    
        if self.t >= self.t_limit:
            done = True    
        if self.game.team_left.life <= 0 or self.game.team_right.life <= 0:
            done = True
            
    
        otherObs = None
        if self.multiagent:
            if self.from_pixels:
                otherObs = cv2.flip(obs, 1) # horizontal flip
            else:
                otherObs = self.game.team_left.getObservations(self.game.ball, self.game.team_right)
    
        info = {
              'ale.lives': self.game.team_right.lives(),
              'ale.otherLives': self.game.team_left.lives(),
              'otherObs': otherObs,
              'state': self.game.team_right.getObservations(self.game.ball, self.game.team_left),
              'otherState': self.game.team_left.getObservations(self.game.ball, self.game.team_right),
              'otherAction': otherActions,  ## the opponent action
              'EnvDepth': self.world.depth
               }
        
                
        """if self.survival_bonus:
              return obs, reward+0.01, done, info"""
         
        return obs, reward , done, [info, copy.copy(info)]

    def init_game_state(self):
        self.t = 0
        self.ret = [0,0]
        self.BallCollision = [0,0]
        self.NotSide = [0,0]
        
        self.game.reset()
        if self.eval_mode:
            self.game.eval_mode = True
        else:
            self.game.eval_mode = False
        if self.training:          ##################################
            self.game.training = True ##################################"
        else:
            self.game.training = False
          
        if not self.update:
            self.world.update = False
            self.world.setup()

    def reset(self):
        if WEBOTS_MODE:
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
      
    def draw_team_name(self):
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
  