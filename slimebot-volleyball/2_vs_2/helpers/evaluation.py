
import typing
from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel


def evaluate_policy(
    model,
    agents,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False
    ):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """

    is_recurrent = False #model.policy.recurrent
    n_agents = 2
    episode_rewards = [[] for i in range(n_agents)]
    episode_lengths = [[] for i in range(n_agents)]
    
    for i in range(n_eval_episodes):
       
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            #print(env.reset())
            observations = env.reset()
            
            # Because recurrent policies need the same observation space during training and evaluation, we need to pad
            # observation to match training shape. See https://github.com/hill-a/stable-baselines/issues/1015
            if is_recurrent:
                
                for ii in range(n_agents):
                    zero_completed_obs = np.zeros((model[ii].n_envs,) + model[ii].observation_space.shape)
                    zero_completed_obs[0, :] = observations[ii]
                    observations[ii] = zero_completed_obs
                    
                    
        done, states = False, [None, None]
        
        episode_reward = [0.0, 0.0]
        episode_length = [0, 0]
        
        while not done:
            
            actions = []
            for i in range(n_agents):
                #action, _,  state, __ = model[i].step(observations[i], state, done)
                action, state = agents[i].predict(observations[i], state=states[i], deterministic=False)
                
                actions.append(action)
                states[i] = state
                
            all_env_action = list(zip(*actions))
                
            new_obs, reward, done, _infos = env.step(all_env_action[0])
            
            if is_recurrent:
                for i in range(n_agents):
                    observations[i][0, :] = new_obs[i]
            else:
                observations[i] = new_obs[i]
                
            episode_reward = [episode_reward[i]+reward[i] for i in range(n_agents)]
            
            if callback is not None:
                callback(locals(), globals())
                
            episode_length = [i+1 for i in episode_length]
            if render:
                env.render()
        for i in range(n_agents):       
            episode_rewards[i].append(episode_reward[i])
            episode_lengths[i].append(episode_length[i])
    
    mean_reward = [] 
    for i in range(n_agents):     
            mean_reward.append(np.mean(episode_rewards[i]))
            
    std_reward = []
    for i in range(n_agents):  
        std_reward.append(np.std(episode_rewards[i]))
        
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
        
    return mean_reward, std_reward
