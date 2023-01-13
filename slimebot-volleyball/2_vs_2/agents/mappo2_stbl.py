import time

import gym
import numpy as np
import tensorflow as tf
import pandas as pd
import os


from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from typing import Union, List, Callable, Optional



class MAPPO2:
    
     
    def __init__(self, policy, env, n_agents, gamma=0.99, n_steps=256, ent_coef=0.01, learning_rate=2.5e-4,
                 vf_coef=0.5,  max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, init_dir = None):
                 
        
        self.policy = policy
        self.env = env
        self.n_agents = n_agents 
        self.gamma  =gamma      
        self.n_steps = n_steps
        self.ent_coef =ent_coef
        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
               
        self.agents = [] 
        self.model_list = []
        self.init_dir = init_dir
        self.set_agents()
        
        runner = Runner(env, self.model_list, n_steps, gamma, lam, self.agents)
        self.runner = runner
        
    def set_agents(self):
        
        for i in range(self.n_agents):
            agent = PPO2(self.policy, self.env, self.gamma, self.n_steps, self.ent_coef, self.learning_rate, 
                         self.vf_coef, self.max_grad_norm, self.lam, self.nminibatches, self.noptepochs,
                         self.cliprange, cliprange_vf=None, verbose=0, tensorboard_log=None,
                         _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                         seed=None, n_cpu_tf_sess=None)
                         
            if self.init_dir is not None:
                print(self.init_dir )
                agent = PPO2.load(self.init_dir+'/init_model', env = agent.env)              
            
            self.agents.append(agent)
            self.model_list.append(agent.act_model)
            
    def load_models(self, path):
        self.agents = []
        for i in range(self.n_agents):
            agent = PPO2.load(path[i]+"/model" + str(i+1)+'/best_model', env = None)  
            
            self.agents.append(agent)
            
    def _init_callback(self,
                      callback: Union[None, Callable, List[BaseCallback], BaseCallback]
                      ) -> BaseCallback:
        """
        :param callback: (Union[None, Callable, List[BaseCallback], BaseCallback])
        :return: (BaseCallback)
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)
        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback
        
        
    
    def learn(self, total_timesteps, callback=None, eval_callback = None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, save_dir = 'MAPPO2'):
        # Transform to callable if needed
        for i in range(self.n_agents):
            self.agents[i].learning_rate = get_schedule_fn(self.agents[i].learning_rate)
            self.agents[i].cliprange = get_schedule_fn(self.agents[i].cliprange)
           
        cliprange_vf_s = [None, None]
        for i in range(self.n_agents):
            cliprange_vf = get_schedule_fn(self.agents[i].cliprange_vf)
            cliprange_vf_s[i] = cliprange_vf
        
        new_tb_log_s = [None, None]
        for i in range(self.n_agents):
            new_tb_log = self.agents[i]._init_num_timesteps(reset_num_timesteps)
            #callback = self.agents[i]._init_callback(callback)
            new_tb_log_s[i] = new_tb_log
        
        callback = self.agents[-1]._init_callback(callback)   
        
        if eval_callback is not None:
            eval_callback.init_callback()         
            
        for i in range(self.n_agents):
            self.agents[i]._setup_learn()

        t_first_start = time.time()
        n_updates = total_timesteps // self.agents[-1].n_batch

        callback.on_training_start(locals(), globals())

        for update in range(1, n_updates + 1):
            assert self.agents[-1].n_batch % self.agents[-1].nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
            batch_size = self.agents[-1].n_batch // self.agents[-1].nminibatches                
            t_start = time.time()
            frac = 1.0 - (update - 1.0) / n_updates
                
                
            lr_now_s = [None, None]
            cliprange_now_s = [None, None]
            cliprange_vf_now_s = [None, None]
            for i in range(self.n_agents):
                lr_now = self.agents[i].learning_rate(frac)
                cliprange_now = self.agents[i].cliprange(frac)
                cliprange_vf_now = cliprange_vf_s[i](frac)
                    
                lr_now_s[i] = lr_now
                cliprange_now_s[i] = cliprange_now
                cliprange_vf_now_s[i] = cliprange_vf_now

            callback.on_rollout_start()
            # true_reward is the reward without discount
            
            
           
            rollout = self.runner.run(eval_callback)
            # Unpack
            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout

            callback.on_rollout_end()

                # Early stopping due to the callback
            if not self.runner.continue_training:
                break
                
            for i in range(self.n_agents):
                self.agents[i].ep_info_buf.extend(ep_infos[i])
                    
            mb_loss_vals = [[] for _ in range(self.n_agents)]
                
            for i in range(self.n_agents):
                if states[i] is None:  # nonrecurrent version
                    update_fac = max(self.agents[i].n_batch // self.agents[i].nminibatches // self.agents[i].noptepochs, 1)
                    inds = np.arange(self.agents[i].n_batch)
                    for epoch_num in range(self.agents[i].noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.agents[i].n_batch, batch_size):
                            timestep = self.agents[i].num_timesteps // update_fac + ((epoch_num *
                                                                                self.agents[i].n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs[i], returns[i], masks[i], actions[i], values[i], neglogpacs[i]))
                            mb_loss_vals[i].append(self.agents[i]._train_step(lr_now_s[i], cliprange_now_s[i], *slices, writer=None,
                                                                     update=timestep, cliprange_vf=cliprange_vf_now_s[i]))

                else:  # recurrent version
                    update_fac = max(self.agents[i].n_batch // self.agents[i].nminibatches // self.agents[i].noptepochs // self.agents[i].n_steps, 1)
                    assert self.agents[i].n_envs % self.agents[i].nminibatches == 0
                    env_indices = np.arange(self.agents[i].n_envs)
                    flat_indices = np.arange(self.agents[i].n_envs * self.agents[i].n_steps).reshape(self.agents[i].n_envs, self.agents[i].n_steps)
                    envs_per_batch = batch_size // self.agents[i].n_steps
                    for epoch_num in range(self.agents[i].noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.agents[i].n_envs, envs_per_batch):
                            timestep = self.agents[i].num_timesteps // update_fac + ((epoch_num *
                                                                                self.agents[i].n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs[i], returns[i], masks[i], actions[i], values[i], neglogpacs[i]))
                            mb_states = states[i][mb_env_inds]
                            mb_loss_vals[i].append(self.agents[i]._train_step(lr_now_s[i], cliprange_now_s[i], *slices, update=timestep,
                                                                     writer=None, states=mb_states,
                                                                     cliprange_vf=cliprange_vf_now_s[i]))

                             
            loss_vals = [np.mean(mb_loss_vals[i], axis=0) for i in range(self.n_agents)]
            t_now = time.time()
            fps = int(self.agents[-1].n_batch / (t_now - t_start))
            writer = None
            if writer is not None:
                total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

            if (update % log_interval == 0 or update == 1):
            #if self.agents[-1].verbose >= 1 and (update % log_interval == 0 or update == 1):
                    
                explained_var = explained_variance(values[0], returns[0])
                logger.logkv("serial_timesteps", update * self.agents[-1].n_steps)
                logger.logkv("n_updates", update)
                logger.logkv("total_timesteps", self.agents[i].num_timesteps)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(explained_var))
                logger.logkv("Environment depth: ", self.env.world.depth)
                    
                if len(self.agents[-1].ep_info_buf) > 0 and len(self.agents[-1].ep_info_buf[0]) > 0:
                    
                    ep_len_mean = safe_mean([ep_info['l'] for ep_info in self.agents[0].ep_info_buf])
                    logger.logkv('ep_len_mean', ep_len_mean)
                    
                    ep_reward_mean = []
                    for i in range(self.n_agents):
                        rew = safe_mean([ep_info['r'] for ep_info in self.agents[i].ep_info_buf])
                        logger.logkv('ep_reward_mean'+str(i), rew)
                        ep_reward_mean.append(rew)
                        
                    ep_ball_collision = []    
                    for i in range(self.n_agents):
                        ball = safe_mean([ep_info['ball_coll'] for ep_info in self.agents[i].ep_info_buf])
                        logger.logkv('ep_ball_collide_mean_'+str(i), ball)
                        ep_ball_collision.append(ball)
                    
                    ep_NotSide = []    
                    for i in range(self.n_agents):
                        notside = safe_mean([ep_info['notSide'] for ep_info in self.agents[i].ep_info_buf])
                        #logger.logkv('ep_not_side_mean_'+str(i), notside)
                        ep_NotSide.append(notside)
                        
                logger.logkv('time_elapsed', t_start - t_first_start)
                    
                for i in range(self.n_agents):
                    for (loss_val, loss_name) in zip(loss_vals[i], self.agents[i].loss_names):
                            logger.logkv(loss_name, loss_val)
                            
                self.save_progress(update, self.agents[0].num_timesteps, ep_len_mean, ep_reward_mean,
                                    self.env.world.depth, ep_ball_collision, ep_NotSide, save_dir)
                logger.dumpkvs()

        callback.on_training_end()
        return self   
    def save_models(self, save_path):
        
        for i in range(len(self.agents)):
            self.agents[i].save(os.path.join(save_path[i], "final_model"+str(i+1)))

    def save_progress(self, update, timestep, mean_len, mean_rew, envDepth, ball_coll, not_side, save_dir = None):
    
        if save_dir is not None and not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        if update == 1:
            data = pd.DataFrame()
            data['total_timesteps'] = []
            data['n_updates'] = []
            data['ep_len_mean'] = []
            data['ep_reward_mean_agt1'] = []
            data['ep_reward_mean_agt2'] = []
            data['Environment Depth']= []
            data['ball_collision_agt1'] = []
            data['ball_collision_agt2'] = []
            data['no_side_agt1'] = []
            data['no_side_agt2'] = []
        else:
            data = pd.read_csv(save_dir +'/progress.csv') 
        
        dico = {'n_updates':update,'total_timesteps':timestep, 'ep_len_mean':round(mean_len,2),
                'ep_reward_mean_agt1':round(mean_rew[0],2), 'ep_reward_mean_agt2':round(mean_rew[1],2),
                'Environment Depth': envDepth, 'ball_collision_agt1': round(ball_coll[0],2),
                'ball_collision_agt2': round(ball_coll[1], 2), 'no_side_agt1': round(not_side[0], 2),
                'no_side_agt2': round(not_side[1],2)
                 }
             
        data = data.append(dico, ignore_index = True)  
        data.to_csv(save_dir + '/progress.csv', index = False)  

class Runner:
    
    def __init__(self, env, model_list, n_steps, gamma, lam, agents):
        """
        A runner to learn the policy of an environment for a model
        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        #super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.env = env
        self.n_envs = env.num_envs
        self.model_list = model_list
        self.n_agents = 2
        self.n_steps = n_steps
        self.states_list = [model.initial_state for model in model_list]
        self.obs_list = self.env.reset()
        print("obs_list", len(self.obs_list), self.obs_list[0].shape)
        self.obs_list = env.reset()
        self.eval_obs_list = env.reset()
        self.eval_states_list = [model.initial_state for model in model_list]
        self.dones = [False for _ in range(self.n_agents)]
        self.eval_dones = [False for _ in range(self.n_agents)]
        
        
        self.agents = agents
        self.continue_training = True

    def run(self, eval_callback):
        """
        Run a learning step of the model
        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [[] for _ in range(self.n_agents)], \
                                                                             [[] for _ in range(self.n_agents)], \
                                                                             [[] for _ in range(self.n_agents)], \
                                                                             [[] for _ in range(self.n_agents)], \
                                                                             [[] for _ in range( self.n_agents)], \
                                                                             [[] for _ in range(self.n_agents)]
        
        
        
        mb_states = self.states_list
        
        epinfos = [[] for _ in range(self.n_agents)]
        
        
        for _ in range(self.n_steps):
        
            
            if eval_callback.check():
                
                self.env.eval_mode = True
                print("Evaluation has started")
                
                eval_epi_rewards, eval_epi_lengths = self.evaluate_policy()
                                                                   
                eval_callback.on_step(eval_epi_rewards, eval_epi_lengths)  
            else:
                eval_callback.on_step(None, None)                                                                                                                                                                                   
                
            
            self.env.eval_mode = False
            all_agent_action = []
            states_list = []
            
            
                  
            for i in range(self.n_agents):                                           
                
                actions, values, states, neglogpacs = self.model_list[i].step(self.obs_list[i], self.states_list[i], self.dones[i])  # pytype: disable=attribute-error
                
                mb_obs[i].append(self.obs_list[i].copy())
                mb_actions[i].append(actions)
                
                all_agent_action.append(actions)
                
                mb_values[i].append(values)
                mb_neglogpacs[i].append(neglogpacs)
                mb_dones[i].append(self.dones[i])
                states_list.append(states)
                            
                     
                #clipped_actions = actions
            # Clip the actions to avoid out of bound error
            #if isinstance(self.env.action_space, gym.spaces.Box):
            #    clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            self.states_list = states_list
            all_env_action = list(zip(*all_agent_action))
            
            #all_env_op_action = list(zip(*all_op_agent_action))
            
            self.obs_list, rewards, self.dones, infos = self.env.step(all_env_action[0])
            
            
            
            #self.other_obs_list = infos[0]['otherState']            
            self.dones = [self.dones for _ in range(2)]
            
            for i in range(self.n_agents):
                self.agents[i].num_timesteps += self.n_envs

            """if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9"""
             
            for i in range(self.n_agents):
                """for info in infos[i]:
                    maybeepinfo = info.get('episode%s' % i)
                    if maybeepinfo: epinfos[i].append(maybeepinfo)"""
                   
                mb_rewards[i].append(rewards[i])
            if self.dones[0] == True:
                for i in range(self.n_agents):
                    epinfos[i].append({'r': self.env.ret[i], 'l': self.env.t, 'ball_coll': self.env.BallCollision[i], 'notSide': self.env.NotSide[i]  })
                    
                #print("new match")
                self.env.reset()
           
        # batch of steps to batch of rollouts
        mb_obs = [np.asarray(obs, dtype=self.obs_list[0].dtype) for obs in mb_obs]
        mb_rewards = [np.asarray(rewards, dtype=np.float32) for rewards in mb_rewards]
        mb_actions = [np.asarray(actions, dtype=np.float32) for actions in mb_actions]
        mb_values = [np.asarray(values, dtype=np.float32) for values in mb_values]
        mb_neglogpacs = [np.asarray(neglogpacs, dtype=np.float32) for neglogpacs in mb_neglogpacs]
        #print(mb_dones[1])
        mb_dones = [np.asarray(dones, dtype=np.bool) for dones in mb_dones]
        mb_dones = [dones.reshape(self.n_steps,1)  for dones in mb_dones]
        
        
        last_values = []

        for i in range(self.n_agents):
            last_values.append(self.model_list[i].value(self.obs_list[i],self.states_list[i],self.dones[i]))
        
        # discount/bootstrap off value fn
        mb_advs = [np.zeros_like(mb_rewards[0]) for _ in range(self.n_agents)]
        
        true_reward = [np.copy(mb_rewards[_]) for _ in range(self.n_agents)]
        
        lastgaelam = [0 for _ in range(self.n_agents)]
        mb_returns = []
        for i in range(self.n_agents):
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - self.dones[i]
                    nextvalues = last_values[i]
                else:
                    nextnonterminal = 1.0 - mb_dones[i][t + 1]
                    nextvalues = mb_values[i][t + 1]

                delta = mb_rewards[i][t] + self.gamma * nextvalues * nextnonterminal - mb_values[i][t]
                mb_advs[i][t] = lastgaelam[i] = delta + self.gamma * self.lam * nextnonterminal * lastgaelam[i]
            # print("mb_values", mb_advs[0].shape,mb_values[0].shape)
            single_returns = mb_advs[i].reshape(self.n_steps, 1) + mb_values[i]

            mb_returns.append(single_returns)
            
        true_reward = [reward.reshape(self.n_steps, 1) for reward in true_reward ]
        
        mb_obs,mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, epinfos, true_reward
        
    def evaluate_policy( self, n_eval_episodes: int = 10, 
                        deterministic: bool = False,callback: Optional[Callable] = None,
                        reward_threshold: Optional[float] = None, 
                        return_episode_rewards: bool = True ):
        
        episode_rewards = [[] for i in range(self.n_agents)]
        episode_lengths = [[] for i in range(self.n_agents)]
        
        for i in range(n_eval_episodes):
           
            
            self.eval_obs_list = self.env.reset()  
     
            self.eval_states_list = [state for state in self.states_list]
            self.eval_dones = [False for _ in range(self.n_agents)]
                                                                     
            
            
            episode_reward = [0.0, 0.0]
            episode_length = [0, 0]
            
            while not self.eval_dones[0]:
                
                actions = []
                states_list = []
                for i in range(self.n_agents):
                    action, _,  state, __ = self.model_list[i].step(self.eval_obs_list[i], self.eval_states_list[i], self.eval_dones[i])
                    #action, state = agents[i].predict(observations[i], state=states[i], deterministic=False)
                    
                    actions.append(action)
                    states_list.append(state)
                
                self.eval_states_list = states_list
                    
                all_env_action = list(zip(*actions))
                self.eval_obs_list, reward, dones, infos = self.env.step(all_env_action[0])
                #new_obs, reward, done, _infos = self.env.step(all_env_action[0])
                self.eval_dones = [dones for _ in range(self.n_agents)]
                
                
                episode_reward = [episode_reward[i]+reward[i] for i in range(self.n_agents)]
                
                if callback is not None:
                    callback(locals(), globals())
                    
                episode_length = [i+1 for i in episode_length]
               
            for i in range(self.n_agents):       
                episode_rewards[i].append(episode_reward[i])
                episode_lengths[i].append(episode_length[i])
        
        mean_reward = [] 
        for i in range(self.n_agents):     
                mean_reward.append(np.mean(episode_rewards[i]))
                
        std_reward = []
        for i in range(self.n_agents):  
            std_reward.append(np.std(episode_rewards[i]))
            
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward, reward_threshold)
        if return_episode_rewards:
            return episode_rewards, episode_lengths
            
        return mean_reward, std_reward        
        



# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr_list):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr_list[0].shape
    #print(s)
    return [arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]) for arr in arr_list]
    
