# Informative path planning toolkit

This tries to be a general set of modular tools which can be composed to easily implement algorithms.

# Installation
If you just want to use the functionality, and it doesn't need to be perfectly up-to-date, pip install is easy. If you want to develop additional functionality or want the latest changes, you should do a local install.

### pip install

You can install a version of this project simply using `pip install ipp-toolkit`. Note that if you install with `pip`, local modifications to your repository won't have any impact.

### local install

Alternatively, for local development, you can use the following instructions.
Begin by installing [anaconda](https://www.anaconda.com/). Create a conda environment called `ipp-toolkit` with `conda create -n ipp-toolkit`.
Activate the environment with `conda activate ipp-toolkit`.
Once this is successful, install `poetry` as described [here](https://python-poetry.org/docs/). This will allow us to install the rest of our dependencies.

Clone this repository and `cd` into it. Now you can install the remaining dependencies with `poetry install`. Now you can try to import this project with `python -c "import ipp_toolkit; print(ipp_toolkit.__version__)"`, which should print the version if successful. Now you can try the notebooks in the `examples` folder.

## DVC

There is some raw data that is managed by [DVC](https://dvc.org/). This stores pointer files to the raw data which is hosted in this [google drive](https://drive.google.com/drive/folders/1P7nJfgDCAHHmpFVRupZxUpy8FGZMy2kd?usp=sharing). Currently, you need to ask me (davidrus@andrew.cmu.edu) for access.

You can download the data using `ipp_toolkit.utils.data.dvc.pull_dvc_data()`. The first time you do this, it will ask you to sign in with the Google account I shared the data to. Give it the requested permissions, and then the download should begin.

## MongoDB
The logging functionality provided by `sacred` requires that you have MongoDB installed on your machine. Instructions can be found [here](https://www.mongodb.com/docs/manual/installation/). Also, you must make sure the service is running, for example with `sudo systemctl start mongod`

# Examples

The level of in-code documentation varies pretty widely in quality. Some functional examples in the form of Jupyter notebooks can be found in the `examples` folder. To run these notebooks in browser, you first need to register the kernel by activating the conda environment and running `ipython kernel install --name "ipp-toolkit" --user`. Then you can run `jupyter notebook`. To run in VSCode, you must select your environment where the `ipp-toolkit` is installed as the kernel. If you have generic issues with Jupyter starting, you may have an issue with which kernel is being used for the Jupyter server. Since everything should be installed in the ipp-toolkit env for the Jupyter server, you can run `Ctrl + Shift + p` and select `Jupyter: Select kernel to Start Jupyter Server` and set it to `ipp-toolkit`.  

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
