# 1. Python

This section shows how to set up python in Webots R2021b using a virtual environment. It may probably not apply to higher versions.

## virtual environment

Once [anaconda](https://docs.anaconda.com/anaconda/install/) is installed, open the command line and enter:
Open the command line and 

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
And any additional library one would like to use ([Tensorflow](https://www.tensorflow.org/install), [Pytorch](https://pytorch.org/get-started/locally/), etc.). The virtual environment can be managed regardless Webots.

