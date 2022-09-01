"""
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


import os
import gym
import numpy as np

from ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from helpers.callbacks import EvalCallback
from helpers.runner import Runner
import pandas as pd

from shutil import copyfile # keep track of generations


from slimevolley3D import Slime3DEnv
# Settings
SEED = 715
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(3e1)
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self
UPGR_LEN = 1500
BEST_LEN = 1000
LOGDIR = "ppo2_selfplay"



class Slime3DEnvSelfPlayEnv(Slime3DEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, realSelf = False):
        super(Slime3DEnvSelfPlayEnv, self).__init__()
        self.realSelf = realSelf
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
                    
        return super(Slime3DEnvSelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
    # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, env, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = BEST_THRESHOLD
        self.upgr_len = UPGR_LEN
        self.n_upgr = 0
        self.count_upgr = 0
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
        if self.last_mean_len >= self.upgr_len:
        
            if self.env.update and (self.n_calls % self.eval_freq == 0):
                self.env.update_world()
                print("SELFPLAY: environment upgraded, actual depth:", int(self.env.world.depth))
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
                                      

def rollout(env, policy):
    """ play one agent vs the other in modified gym-style loop. """
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _states = policy.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
        if RENDER_MODE:
              env.render()

    return total_reward
  
class Baseline:
    def __init__(self, policy):
        self.policy = policy
    
    def predict(self, obs):
        action, _ = self.policy .predict(obs)
        return action
    

def train():
    # train selfplay agent
    logger.configure(folder=LOGDIR)

    env = Slime3DEnvSelfPlayEnv() 
    env.training = True
    env.update = True
    env.world.setup(n_update = 4, init_depth = 0)
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

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()

if __name__=="__main__":

    train()