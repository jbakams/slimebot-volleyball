import sys
main_path = "/home/jey/Documents/GitHub/slimebot-volleyball/slimebot-volleyball"
sys.path.append(main_path)



import os
import gym
import numpy as np


from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from helpers.callbacks import EvalCallback
from helpers.runner import Runner

from controllers.selfplay_training_ppo.ppo2 import PPO2
import pandas as pd

from shutil import copyfile # keep track of generations
from environments.volleybot import VolleyBotEnv

SEED = 7542
LOGDIR = "/controllers/selfplay_training_ppo/ppo2_selfplay"

def rollout(env, policy1, policy2 = None):
    """ play one agent vs the other in modified gym-style loop. """
    obs1 = env.reset()
    if policy2 is not None:
        obs2 = obs1

    done = False
    total_reward = 0
    
    if policy2 is None:

        while not done:
    
            action1, _states1 = policy1.predict(obs1)
            obs1, reward, done, _ = env.step(action1)
            total_reward += reward            
    
        return total_reward
        
    else:
        while not done:
    
            action1, _states1 = policy1.predict(obs1)
            action2, _states2 = policy1.predict(obs2)
            obs1, reward, done, _ = env.step(action1, action2)
            obs2 = _["otherObs"]
            total_reward += reward
            
        return total_reward
         
 
class Baseline:
    def __init__(self, policy):
        self.policy = policy
    
    def predict(self, obs):
        action, _ = self.policy .predict(obs)
        return action
    

def evaluate():
    

    env = VolleyBotEnv()    
    env.world.setup(init_depth = 24)
    env.seed(SEED)
  

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    model = PPO2(MlpPolicy, env, n_steps=4096, cliprange=0.2, ent_coef=0.0, noptepochs=4,
                   learning_rate=3e-4, nminibatches=64, gamma=0.99, lam=0.95, verbose=2)
                   
    model = PPO2.load(main_path + LOGDIR + "/best_model", env=None)
    
    env.policy = Baseline(model)
    
    history = []
  
    for i in range(1000):
        score = rollout(env, model)
        print("Match "+ str(i+1) + " score: ", score)
        history.append(score)

if __name__=="__main__":

    evaluate()