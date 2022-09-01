# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This controller gives to its robot the following behavior:
According to the messages it receives, the robot change its
behavior.
"""
from controller import Robot

class AgentRobot(Robot):

    timeStep = 32
    
    def __init__(self):
        super(AgentRobot, self).__init__()
        self.camera = self.getDevice('camera')
        self.camera.enable(4 * self.timeStep)
        #self.emitter = self.getDevice('emitter')       

    def getobs(self):
        return self.camera.getImageArray()
        
    def run(self):
        while True:
            #self.getobs()
            self.step(self.timeStep)          
                      

controller = AgentRobot()
controller.run()
