import os
import gym
import numpy as np

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from helpers.callbacks import EvalCallback

from shutil import copyfile # keep track of generations

from slimevolley3D import Slime3DEnv
# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(3e1)
BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self
UPGR_LEN = 2000
BEST_LEN = 1000

RENDER_MODE = False # set this to false if you plan on running for full 1000 trials.

LOGDIR = "ppo1_selfplay"

class Slime3DEnvSelfPlayEnv(Slime3DEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self):
    super(Slime3DEnvSelfPlayEnv, self).__init__()
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
        self.best_model = PPO1.load(filename, env=self)
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
    
  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    #self.n_upgr = self.n_calls// self.eval_freq
          
    """if result and (self.best_mean_len == self.prev_best_mean_len and self.last_mean_reward  > self.best_mean_reward):
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.last_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = self.last_mean_reward"""
      
    if result and self.best_mean_len > self.prev_best_mean_len:
      self.generation += 1
      print("SELFPLAY: mean_lenght achieved:", self.best_mean_len)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.prev_best_mean_len = self.best_mean_len
    
    #print(self.env.game.agent_right.getObservation()  )  
    if self.last_mean_len >= self.upgr_len:
    
      if self.env.upgrade and (self.n_calls % self.eval_freq == 0):
        self.env.upgrade_world()
        print("SELFPLAY: environment upgraded, actual depth:", self.env.world.depth)
        self.prev_best_mean_len = self.best_mean_len = BEST_LEN  
        
      
      
                                
    return result

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

def train():
  # train selfplay agent
  logger.configure(folder=LOGDIR)

  env = Slime3DEnvSelfPlayEnv() 
  env.training = True
  env.upgrade = True
  env.world.setup(n_upgr = 6, init_depth = 0)
  env.seed(SEED)

  # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
  model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=4,
                   optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)

  eval_callback = SelfPlayCallback(env, env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

  env.close()

if __name__=="__main__":

  train()