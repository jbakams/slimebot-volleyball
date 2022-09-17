# 1. Python

This section shows how to set up python in Webots R2021b using a virtual environment. It may probably not apply to higher versions.

## virtual environment

Once [anaconda](https://docs.anaconda.com/anaconda/install/) is installed, open the command line and enter:

```cmd
conda create -n webots python=3.7
conda activate webots
```
## Dependencies

With the virtual env activated, install all dependencies using pip

```cmd
pip install --upgrade pip
pip install gym
pip install opencv-python
```
And any additional library one would like to use ([Tensorflow](https://www.tensorflow.org/install), [Pytorch](https://pytorch.org/get-started/locally/), etc.).

## Python path in Webots R2021b

Open Webots. On the top right go to the follwoing options: *Tools>Preferences*

On the Preferences window in the *python command* line set the path the python of the virtual environment
<p align="center">
  <img width="40%" src="https://github.com/jbakambana/slimebot-volleyball/blob/main/Images/ref1.png"></img>
</p>

Hence, we are all set. The virtual environment can be managed regardless of Webots. It can be used for any other projects apart from those using Webots. To use another virtual environment, just change the path.

# 2. Webots

[Webots](https://cyberbotics.com/) is multi-purposes software for robotic simualtions. You can refer to the [User Guide](https://cyberbotics.com/doc/guide/getting-started-with-webots) to have an overview on how it works. In this section we are explaining the important options regarding our projects.

## Interface overview

<p align="center">
  <img width="45%" src="https://github.com/jbakambana/slimebot-volleyball/blob/main/Images/ref2.png"></img>
</p>
<p align="center">
 Typical Webots interface
</p>

According to numbers in the image above:

1. 3D scene: Serves as a realtime visualizer of the virtual environment.
2. Console: Webots has an built-in console that help to follow up progress of simulations.
3. IDE: Webots has a built-in IDE that allows to type codes and scripts directly in Webots. It is possible to use an external IDE such Visual Studio or Pycharm using the following [guide](https://cyberbotics.com/doc/guide/using-your-ide) to run simulation outside of Webots. 
4. Scene Tree: The scene tree contain the list of all elements in the 3D scene and their specific attributes (color, physics, etc.) For this game we don't have to worry about it, except the need of customizing the default 3D scene of the game.

## Running a simulation

There are some Buttons on top of the 3D scene and Scene Tree pannels. Place the cursor on top of a button and a text message will appear to explain what is the button use for. Let give a briefing about some of them.

<p align="center">
  <img width="45%" src="https://github.com/jbakambana/slimebot-volleyball/blob/main/Images/ref3.png"></img>
</p>

1. Restore the simulation at the initial state. Click on it only if you need to restart a simulation (For examlple after fixing bugs) otherwise you may lose all progress so far.
2. Execute simulation for only one timestep. May be helpful to check if everything is running accordingly in the 3D scene.
3. Run the simulation in real-time. If you want to enjoy the simulation in real time than there is your button.
4. Run the simulation as fast as possible. There is the button you need when training agents. The speed depends on the capicity of the used equipment.
5. Hide or Show rendering. The rendering use graphic and may slow down a bit the simulation, even in the fast speed mode. You can hide the 3D scene by clicking on it to add more speed to the simulation.
6. Record the simulation as a video to save locally in your device.

The timeline pannel on the left of the show the running time of the simulation. you can pause the simulation and continue it later, as long as Webots doesn't shut down.

## Open a project

To open a project in webots go on the top right a follow those options: *File>Open World*

Then navigate into your device where the prject is saved and seek for the file: *world.wbt*. If you cloned this repository this should be the path to the world.wbt file: 

  */home/usrname/slimebot-volleyball/slimebot-volleyball/worlds/world.wbt*
  
 Once the project open, you can run the simulation. But should make sure to have the right controller script running.

## Controllers

A [controller](https://cyberbotics.com/doc/guide/controller-programming) in webots is a script in charge on controller dynamics of the simulation. They are two kind of controllers *robot-controller* and *supervisor-controller*. We are more interested in the supervisor for now. To train or evaluate an agent we need to define the script as a controller such that *Webots* will recognize it.

To use a controller: On the Scene Tree unroll the arrow right before *DEF SCENE_CONTROL Robot*

<p align="center">
  <img width="25%" src="https://github.com/jbakambana/slimebot-volleyball/blob/main/Images/scenecontroller.png"></img>
</p>
