"""
Running a random agent 
"""

import sys
sys.path.append("/home/jey/Documents/GitHub/slimebot-volleyball/slimebot-volleyball")
from environments.volleybot import VolleyBotEnv

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    
    def predict(self):
        action  = self.env.action_space.sample()
        return action
    

def play(n_episodes = 1000):
    """ play the random agent vs the built-in env random agnet""" 
    
    for i in range(n_episodes):
        env = VolleyBotEnv()    
        env.world.setup(init_depth = 4)
        env.seed(157)
        
        agent = RandomPolicy(env)
  
    
        done = False
        while not done:
    
            action = agent.predict()
            obs, reward, done, _ = env.step(action)
            

if __name__=="__main__":

    play()