# Informative path planning toolkit
This tries to be a general set of modular tools which can be composed to easily implement algorithms.

# Installation
Begin by installing [anaconda](https://www.anaconda.com/). Create a conda environment called `ipp-toolkit` with `conda create -n ipp-toolkit`. 
Activate the environment with `conda activate ipp-toolkit`. Now install pytorch into this environment as described [here](https://pytorch.org/get-started/locally/).
Once this is successful, install `poetry` as described [here](https://python-poetry.org/docs/). This will allow us to install the rest of our dependencies.

Clone this repository and `cd` into it. Now you can install the remaining dependencies with `poetry install --dev`.
Now you can try running `python dev/experiments/simple.py`

`cd` into the gym-ipp directroy.  Now run `pip install -e .` to install the gym environments.  You should now be able to run `python dev/experiments/simple_gym.py`

## DVC
There is some raw data which is managed by [DVC](https://dvc.org/). This stores pointer files to the raw data which is hosted elsewhere. Unfortunately, most of the data cannot be shared outside of the lab, which means only some people have access to the remote data store. If needed, I will try to implement tiers of data access. 

# Structure
The code lives in `ipp_toolkit`. The main elements are the following:
* **data**: Wraps a real or simulated dataset. Returns a value at location. 
* **sensors**: Takes a dataset at initialization. Then returns a value at a given location but can induce noise or other artifacts.
* **world_models**: Represents our current belief about the world. Can be updated with new observations
* **planners**: Takes a world model and determines where to sample next.
* **utils**: General code utilities. 
* **visualization**:  Currently unused, most visualization is handled by individual modules
