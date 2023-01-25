# Informative path planning toolkit
This tries to be a general set of modular tools which can be composed to easily implement algorithms.

# Installation
The majority of this project is easy to install but there are some headaches around `pytorch`. The current approach I'm taking is to have you install `pytorch` and the libraries that directly depend on it on your own. This means you can install pytorch using your favorite method, e.g. from [here](https://pytorch.org/get-started/locally/). Then you should install `gpytorch` and `torchgeo` with pip. Then you can install this package and the the other dependencies using poetry or pip. 

### pip install
This project can now be pip-installed, but it is not fully tested. This requires `python>=3.9`.  
```pip install ipp-toolkit```. You can use the jupyter notebooks once you've installed the project from pip, but note the edits to the local files will not have any impact.

### local install
Alternatively, for local development, you can use the following instructions.
Begin by installing [anaconda](https://www.anaconda.com/). Create a conda environment called `ipp-toolkit` with `conda create -n ipp-toolkit`. 
Activate the environment with `conda activate ipp-toolkit`.
Once this is successful, install `poetry` as described [here](https://python-poetry.org/docs/). This will allow us to install the rest of our dependencies.

Clone this repository and `cd` into it. Now you can install the remaining dependencies with `poetry install`.
Now you can try running `python dev/experiments/simple.py`

`cd` into the gym-ipp directroy.  Now run `pip install -e .` to install the gym environments.  You should now be able to run `python dev/experiments/simple_gym.py`

## DVC
There is some raw data which is managed by [DVC](https://dvc.org/). This stores pointer files to the raw data which is hosted in this [google drive](https://drive.google.com/drive/folders/1P7nJfgDCAHHmpFVRupZxUpy8FGZMy2kd?usp=sharing). Currently you need to ask me (davidrus@andrew.cmu.edu) for access.

You can download the data using `ipp_toolkit.utils.data.dvc.pull_dvc_data()`. The first time you do this, it will ask you to sign in with the google account I shared the data to. Give it the requested permissions, and then download should begin. 

# Usage
The level of in-code documentation varies pretty widely in quality. Some functional examples in the form of jupyter notebooks can be found in the `examples` folder.

# Structure
The code lives in `ipp_toolkit`. The main elements are the following:
* **data**: Wraps a real or simulated dataset. Returns a value at location. 
* **sensors**: Takes a dataset at initialization. Then returns a value at a given location but can induce noise or other artifacts.
* **world_models**: Represents our current belief about the world. Can be updated with new observations
* **planners**: Takes a world model and determines where to sample next.
* **predictors**: Predict some target quantity, such as the geo-spatial quantity of interest.
* **utils**: General code utilities. 
* **visualization**:  Currently unused, most visualization is handled by individual modules
* **experiments**: System-level evaluation to answer a question.

Additional modules are being depricated:
* **world_models**: Are being refactored into predictors
* **trainers**: Should be put into a utility, since they are for only reinforcement learning.
