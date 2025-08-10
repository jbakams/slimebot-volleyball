import sys
main_path = "/home/usr/Documents/GitHub/slimebot-volleyball/slimebot-volleyball/2_vs_2"
sys.path.append(main_path)

import os
import gym
import numpy as np
import pandas as pd

from agents.mappo2_stbl import MAPPO2
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from helpers.callbacks import EvalCallback
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv

from shutil import copyfile # keep track of generations
from environments.team_volleybot import TeamVolleyBot

from helpers.evaluation import evaluate_policy

#from multi_env_tools.vec_normalize import VecNormalize
#from multi_env_tools.subpro_vec_env import SubprocVecEnv
#from multi_env_tools.vec_monitor import VecMonitor
import datetime

  
SEED = 392

NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(3e1)
BEST_THRESHOLD = 0. # must achieve a mean score above this to replace prev best self
UPGR_LEN = 1100
BEST_LEN = 1000

RENDER_MODE = False # set this to false if you plan on running for full 1000 trials.

LOGDIR = ["MAPPO2_"+str(SEED)+"/model1", "MAPPO2_"+str(SEED)+"/model2"]

for i in range(2):
    if not os.path.exists(LOGDIR[i]):               
        os.makedirs(LOGDIR[i])
        
logdir = ["MAPPO2_"+str(SEED)+"/model1", "MAPPO2_"+str(SEED)+"/model2"]
SAVE_DIR = 'MAPPO2_'+ str(SEED)
INIT_DIR = "MAPPO2_"+str(SEED)+"/init_model"

class TeamVolleyBotSelfPlayEnv(TeamVolleyBot):
  # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, n_agents = 2):
        super(TeamVolleyBotSelfPlayEnv, self).__init__()
        self.policy = self
        self.best_model = [None, None]
        self.best_model_filename = [None, None]
        self.n_agents = n_agents
        
    def predict(self, obs): # the policy
        
        if self.best_model[0] is None:
            
            return [self.action_space.sample(), self.action_space.sample()]  # return random actions
            
        else:
            actions = []
            for i in range(self.n_agents):
                action, _ = self.best_model[i].predict(obs[i])
                
                actions.append(action[0])
                
            return actions
      
    def reset(self):
        # load model if it's there
        # bestmodels = []
        for i in range(self.n_agents):
                
            modellist = [f for f in os.listdir(LOGDIR[i]) if f.startswith("history")]
            modellist.sort()
            if len(modellist) > 0:
                filename = os.path.join(LOGDIR[i], modellist[-1]) # the latest best model
                if filename != self.best_model_filename[i]:
                    print("loading model"+str(i)+" : ", filename)
                    self.best_model_filename[i] = filename
                    #if self.best_model[i] is not None:
                        #self.best_model[i] = None
                    self.best_model[i] = PPO2.load(filename, env=None)
        return super(TeamVolleyBotSelfPlayEnv, self).reset()
        
        
class SelfPlayCallback:
    # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, 
          env,
          models,
          n_eval: int = 5,
          eval_freq: int = 10000,
          n_eval_episodes: int = 5,
          log_path: list = [None, None],
          best_model_save_path: list = [None, None],
          deterministic: bool = False,
          render: bool = False,
          verbose: int = 1):       
          
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.best_mean_len = -np.inf
        self.last_mean_len = -np.inf
        self.n_agents = env.n_agents
        self.n_calls = 0
        self.verbose = 1
        self.models = models
        
        self.best_mean_reward = BEST_THRESHOLD
        self.upgr_len = UPGR_LEN
        self.n_upgr = 0
        self.count_upgr = 0
        self.generation = 0
        
        self.best_mean_len = BEST_LEN
        self.prev_best_mean_len = BEST_LEN
        
        self.new_model = False
        self.new_depth = False
        
        
        assert env.num_envs == 1, "You must pass only one environment for evaluation"

        self.best_model_save_path = best_model_save_path
        
        # Logs will be written in `evaluations.npz`
        for i in range(self.n_agents ):
            if log_path[i] is not None:
                log_path[i] = os.path.join(log_path[i], 'evaluations')
            
        
        if not isinstance(env, VecEnv):
            eval_env = DummyVecEnv([lambda: env])
            
        #self.eval_env = eval_env
        
                
        self.log_path = log_path
        self.evaluations_results = [[] for i in range(self.n_agents )]
        self.evaluations_timesteps = [[] for i in range(self.n_agents)]
        self.evaluations_length = [[] for i in range(self.n_agents)]

    def init_callback(self):
        
        # Create folders if needed
        for i in range(self.n_agents):
            if self.best_model_save_path[i] is not None:
                os.makedirs(self.best_model_save_path[i], exist_ok=True)
            if self.log_path[i] is not None:
                os.makedirs(os.path.dirname(self.log_path[i]), exist_ok=True)
    
    def check(self):
        
        self.n_calls += 1
        
        return self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        
                
    def on_step(self, episode_rewards, episode_lengths) -> bool:
        
        if episode_rewards is None:
            self.env_progress(self.env.world.depth)
            return True
                                                                   
        for i in range(self.n_agents):
            if self.log_path[i] is not None:
                self.evaluations_timesteps[i].append(self.models.agents[i].num_timesteps)
                self.evaluations_results[i].append(episode_rewards[i])
                self.evaluations_length[i].append(episode_lengths[i])
                np.savez(self.log_path[i], timesteps=self.evaluations_timesteps[i],
                             results=self.evaluations_results[i], ep_lengths=self.evaluations_length[i])
                         
                        
        assert episode_rewards[0] == episode_rewards[1], "Rewards should be equal during evaluation"
               
        mean_reward, std_reward = np.mean(episode_rewards[0]), np.std(episode_rewards[0])
        mean_ep_length, std_ep_length = np.mean(episode_lengths[0]), np.std(episode_lengths[0])
            # Keep track of the last evaluation, useful for classes that derive from this callback
        self.last_mean_reward = mean_reward
        self.last_mean_len  = mean_ep_length
	    
        if self.verbose > 0:
            print("Eval num_timesteps={}, "
                  "episode_reward={:.2f} +/- {:.2f}".format(self.models.agents[0].num_timesteps, mean_reward, std_reward))
            print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))           
                    
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
                    
            for i in range(self.n_agents):
                if self.best_model_save_path[i] is not None:
                    self.models.agents[i].save(os.path.join(self.best_model_save_path[i], 'best_model'))
            self.best_mean_reward = mean_reward
            # Trigger callback if needed
            """if self.callback is not None:
                    return self._on_event()"""
        
        if  self.best_mean_reward > BEST_THRESHOLD:
            self.generation += 1
            print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
            print("SELFPLAY: new best model, bumping up generation to", self.generation)
            for i in range(self.n_agents):
                source_file = os.path.join(LOGDIR[i], "best_model.zip")
                backup_file = os.path.join(LOGDIR[i], "history_"+str(self.generation).zfill(8)+".zip")
                copyfile(source_file, backup_file)
            self.best_mean_reward = BEST_THRESHOLD
            self.new_model = True 
        #return result                                  
 
        if self.last_mean_len >= self.upgr_len:
    
            if self.env.update and (self.n_calls % self.eval_freq == 0):
                self.env.update_world()
                
                print("SELFPLAY: environment updated, actual depth:", int(self.env.world.depth))
                self.new_depth = True 
                for i in range(self.n_agents):
                    self.models.agents[i].save(os.path.join(self.best_model_save_path[i],"increment_model_"+str(int(self.env.world.depth)).zfill(4)+".zip"))
                if mean_reward < BEST_THRESHOLD:
                    self.generation += 1
                    for i in range(self.n_agents):
                        self.models.agents[i].save(os.path.join(self.best_model_save_path[i],"history_"+str(self.generation).zfill(8)+".zip"))
                #self.prev_best_mean_len = self.best_mean_len = BEST_LEN  
        self.env_progress(self.env.world.depth) 
                             
                                
        return True

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
                data = pd.read_csv(SAVE_DIR +'/env.csv')    
           
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
        
            data.to_csv(SAVE_DIR + '/env.csv', index = False)
  
  
def Train():
    
    env = TeamVolleyBotSelfPlayEnv() 
    env.training = True
    env.update = True
    #env.world.stuck = True
    env.world.setup(n_update = 24, init_depth = 6)
    env.seed(SEED)
    n_agents = 2
    total_timesteps = int(1e9)
    
    models = MAPPO2(MlpPolicy, env, n_agents, gamma=0.99, n_steps=4096, ent_coef=0.01, learning_rate=3e-4,
                 vf_coef=0.5,  max_grad_norm=0.5, lam=0.95, nminibatches=64, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, verbose=2, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, init_dir = None)
    #env.atari_mode = True
    
    
    eval_callback = SelfPlayCallback(env = env,
                                     models = models,
                                     best_model_save_path=LOGDIR,
                                     log_path=logdir,
                                     eval_freq=EVAL_FREQ,
                                     n_eval_episodes=EVAL_EPISODES,
                                     deterministic=False                                    
                                     )

    models.learn(total_timesteps, eval_callback = eval_callback, save_dir = SAVE_DIR)
    models.save_models(LOGDIR)

if __name__=="__main__":

    Train()
