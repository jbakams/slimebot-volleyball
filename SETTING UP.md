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

Hence, we are all set. The virtual environment can be managed regardless of Webots. It can be used for any other projects apart from those using Webots. To use another virtual environment, just change the path to another one.







