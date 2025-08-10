import sys

## Main path if using Linux
#main_path = "/home/usr/Documents/GitHub/slimebot-volleyball/slimebot-volleyball/2_vs_2"

## Main path is using windows
main_path ="C:\\Users\\usr\\Documents\\GitHub\\slimebot-volleyball\\slimebot-volleyball\\2_vs_2"
sys.path.append(main_path)

import os


import os
import gym
import numpy as np


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from agents.mappo2_stbl import MAPPO2 

import pandas as pd

from shutil import copyfile # keep track of generations


from environments.team_volleybot import TeamVolleyBot


SEED = 425

## Directory path for trained models if using Linux
#dir = "/trained_agents/MAPPO2_425"

## Direcory path for trained models if using windows
dir = "\\trained_agents\\MAPPO2_425"
LOGDIR = [dir, dir]

def rollout(env, models1, models2 = None):
    """ play one agent vs the other in modified gym-style loop. """
    obs1 = env.reset()
    if models2 is not None:
        obs2 = obs1

    done = False
    total_reward = 0
    
    if models2 is None:

        while not done:
    
            action1_1, _ = models1[0].predict(obs1[0])
            action1_2, _ = models1[1].predict(obs1[1])
            
            obs1, reward, done, info = env.step([action1_1[0], action1_2[0]])
            total_reward += reward[0]         
    
        return total_reward
        
    else:
        while not done:
    
            action1_1, _ = models1[0].predict(obs1[0])
            action1_2, _ = models1[1].predict(obs1[1])
            
            action2_1, _ = models2[0].predict(obs2[0])
            action2_2, _ = models2[1].predict(obs2[1])
            
            obs1, reward, done, info = env.step(action1)
            total_reward += reward    
            
            obs2 = info[0]["otherObs"]    
            
        return total_reward
         
 
class Baseline:
    def __init__(self, policy):
        self.policy = policy
    
    def predict(self, obs):
        actions = []
        for i in range(2):
            action, _ = self.policy[i].predict(obs[i])
            actions.append(action[0])
        #print(actions)
        return actions
    

def evaluate():
    

    env = TeamVolleyBot()  
    env.update = True  
    env.world.setup(n_update = 6, init_depth = 24)
    env.eval_mode = True
    env.training = False
    env.seed(SEED)
  

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    models = MAPPO2(MlpPolicy, env, n_agents = 2, gamma=0.99, n_steps=4096, ent_coef=0.01, learning_rate=3e-4,
                 vf_coef=0.5,  max_grad_norm=0.5, lam=0.95, nminibatches=64, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, verbose=2, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, init_dir = None)
    
    #print(type(models))               
    models.load_models([main_path + LOGDIR[i] for i in range(2)] )
    #
    env.policy = Baseline(models.agents)
    
    history = []
  
    for i in range(1000):
        print("Evaluated in environment depth :", env.world.depth)
        score = rollout(env, models.agents)
        print("Match "+ str(i+1) + " score: ", score)
        history.append(score)

if __name__=="__main__":
     evaluate()
