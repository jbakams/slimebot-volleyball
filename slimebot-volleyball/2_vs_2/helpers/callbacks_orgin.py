
import os
from abc import ABC
import warnings
import typing
from typing import Union, List, Dict, Any, Optional

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from helpers.evaluation import evaluate_policy
from stable_baselines import logger

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error

from stable_baselines.common.callbacks import *

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: Optional[str] = None,
                 best_model_save_path: Optional[str] = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
                 
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.best_mean_len = -np.inf
        self.last_mean_len = -np.inf
        self.n_agents = eval_env.n_agents

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        for i in range(self.n_agents ):
            if log_path[i] is not None:
                log_path[i] = os.path.join(log_path[i], 'evaluations')
            
                
        self.log_path = log_path
        self.evaluations_results = [[] for i in range(self.n_agents )]
        self.evaluations_timesteps = [[] for i in range(self.n_agents)]
        self.evaluations_length = [[] for i in range(self.n_agents)]

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))
        
        # Create folders if needed
        for i in range(self.n_agents):
            if self.best_model_save_path[i] is not None:
                os.makedirs(self.best_model_save_path[i], exist_ok=True)
            if self.log_path[i] is not None:
                os.makedirs(os.path.dirname(self.log_path[i]), exist_ok=True)
        
        
    def _on_step(self) -> bool:
    
        print("a")
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)
            
            self.eval_env.eval_mode = True
            Print("Evakuation started")
            episode_rewards, episode_lengths = evaluate_policy(self.model_list, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)
            self.eval_env.eval_mode = False
            
            
            for i in ragne(self.eval_env.n_agents):
                if self.log_path[i] is not None:
                    self.evaluations_timesteps[i].append(self.num_timesteps)
                    self.evaluations_results[i].append(episode_rewards[i])
                    self.evaluations_length[i].append(episode_lengths[i])
                    np.savez(self.log_path[i], timesteps=self.evaluations_timesteps[i],
                             results=self.evaluations_results[i], ep_lengths=self.evaluations_length[i])
                         
                         
            assert episode_rewards[0] != episode_rewards[1], "Rewards should be equal during evaluation"
               
            mean_reward, std_reward = np.mean(episode_rewards[0]), np.std(episode_rewards[0])
            mean_ep_length, std_ep_length = np.mean(episode_lengths[0]), np.std(episode_lengths[0])
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward
            self.last_mean_len  = mean_ep_length
	    
            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))           
                    
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                    
                for i in range(self.n_agnets):
                    if self.best_model_save_path[i] is not None:
                        self.model_list[i].save(os.path.join(self.best_model_save_path[i], 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
                    
        return True
