"""


Training stablebaselines ppo2 in a selfplay fashion

Original script:
https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py

Simulator:
https://github.com/cyberbotics/webots

At the moment the script is set to run only in Webots.

There is some modifications comparing to the original script from the slimevolleygym. 
The selfplay mode can be done either by playing against the best previous evaluated copy of the 
learner (slimevolleygym approach) or it can be done by playing instantly with the current version
of the learner.
"""

import sys 

#appending the main game directory path

#####    Linux   #######
#sys.path.append("/home/jey/Documents/GitHub/slimebot-volleyball/slimebot-volleyball/1_vs_1")

#####    Windows    ######
sys.path.append("C:\Users\usr\Documents\GitHub\slimebot-volleyball\slimebot-volleyball\1_vs_1")



import os
import gym
import numpy as np

from ppo2 import PPO2 # import the customized stablebaselines ppo
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from helpers.callbacks import EvalCallback
import pandas as pd
from shutil import copyfile # keep track of generations


from environments.volleybot import VolleyBotEnv
# Settings
SEED = 723 # Try different seed 
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self
INCREMENT_THRESHOLD = 1500 # Increase the depth each time the agent reaches this average episode lenght during evaluation
BEST_LEN = 1500

LOGDIR = "ppo2_selfplay"



class VolleyBotSelfPlayEnv(VolleyBotEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, realSelf = False):
        super(VolleyBotSelfPlayEnv, self).__init__()
        self.realSelf = realSelf # tells if the opponent will be the real time version of the trainer
        if not self.realSelf:       
            self.policy = self
        self.best_model = None
        self.best_model_filename = None
        
        
    def predict(self, obs): # the policy
        if self.best_model is None:
            return self.action_space.sample() # return a random action         
        else:
            action, _ = self.best_model.predict(obs)
            return action
            
    def reset(self):
        if not self.realSelf:
            # load model if it's there         
            modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
            modellist.sort()
            if len(modellist) > 0:
                filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
                if filename != self.best_model_filename:
                    print("loading model: ", filename)
                    self.best_model_filename = filename
                    if self.best_model is not None:
                        del self.best_model
                    self.best_model = PPO2.load(filename, env=None)
                    
        return super(VolleyBotSelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
    # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, env, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        
        self.best_mean_reward = BEST_THRESHOLD
        self.increment_threshold = INCREMENT_THRESHOLD
        self.n_increment = 0
        self.count_increment = 0
        self.generation = 0
        self.env = env
        self.best_mean_len = BEST_LEN
        self.prev_best_mean_len = BEST_LEN
        self.prevEnvDepth = 0
        self.new_model = False
        self.new_depth = False
    
    def _on_step(self) -> bool:
       
        result = super(SelfPlayCallback, self)._on_step()
        
        if result and self.best_mean_reward > BEST_THRESHOLD:
            self.generation += 1
            print("SELFPLAY: mean_reward achiaeved:", self.best_mean_reward)
            print("SELFPLAY: new best model, bumping up generation to", self.generation)
            source_file = os.path.join(LOGDIR, "best_model.zip")
            backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
            copyfile(source_file, backup_file)
            self.best_mean_reward = BEST_THRESHOLD  
            self.new_model = True                       
    
        #print(self.env.game.agent_right.getObservation()  )  
        if self.last_mean_len >= self.increment_threshold:
        
            if self.env.update and (self.n_calls % self.eval_freq == 0):
                self.env.increment_world()
                print("SELFPLAY: Environment incremented, actual depth:", int(self.env.world.depth))
                self.prev_best_mean_len = self.best_mean_len = BEST_LEN 
                self.new_depth = True               
            
        self.env_progress(self.env.world.depth) 
        
        return result
        
    def env_progress(self, depth):
        """
        Function that keep track on the environment dynamics. Register each training timestep in which
        a new opponent is saved or in which the depth of the environment is incremented. And sae the 
        progress as csv file
        
        """     
        if self.n_calls % 4096 == 0:        
            if self.n_calls // 4096 == 1:
                data = pd.DataFrame()
                data['depth'] = []
                data['opponent'] = []
            else:
                data = pd.read_csv(LOGDIR +'/env.csv')    
           
            if self.new_depth: # If the depth is incremented
                row1 = self.n_calls        
                self.new_depth = False
            else:
                row1 = 0
            
            if self.new_model: # If a new model is saved
                row2 =  self.n_calls
                self.new_model = False
            else:
                row2 = 0               
        
            dico = {'depth': row1 , 'opponent' : row2}
            data = data.append(dico, ignore_index = True)
        
            data.to_csv(LOGDIR + '/env.csv', index = False)
                                      
  
class Baseline:
    def __init__(self, policy):
        self.policy = policy
    
    def predict(self, obs):
        action, _ = self.policy .predict(obs)
        return action
    

def train():
    # train selfplay agent
    logger.configure(folder=LOGDIR)

    env = VolleyBotSelfPlayEnv() 
    env.training = True
    env.update = True
    env.world.stuck = False
    env.world.setup(n_increment = 6, init_depth = 0)
    env.seed(SEED)
  

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    model = PPO2(MlpPolicy, env, n_steps=4096, cliprange=0.2, ent_coef=0.0, noptepochs=4,
                   learning_rate=3e-4, nminibatches=64, gamma=0.99, lam=0.95, verbose=2)
                   
  

    eval_callback = SelfPlayCallback(env, env,
                    best_model_save_path=LOGDIR,
                    log_path=LOGDIR,
                    eval_freq=EVAL_FREQ,
                    n_eval_episodes=EVAL_EPISODES,
                    deterministic=False)
    
    # Uncomment the 2 next lines to use a real time selfplay
    #env.realSelf = True
    #env.policy = Baseline(model)
  
    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(LOGDIR, "final_model"))
    env.close()

if __name__=="__main__":

    train()
