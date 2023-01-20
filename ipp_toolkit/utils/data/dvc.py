import os


def pull_dvc_data():
    os.system("dvc pull")


def push_dvc_data():
    os.system("dvc push")
