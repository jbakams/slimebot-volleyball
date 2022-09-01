"""
The following controller manage the camera view of the robot
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
