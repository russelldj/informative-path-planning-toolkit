import os


def pull_dvc_data(folder: str = None):
    """Can optionally specify a folder to pull"""
    if folder is None:
        os.system("dvc pull")
    else:
        os.system(f"dvc pull -R {folder}")


def push_dvc_data(folder: str = None):
    if folder is None:
        os.system("dvc push")
    else:
        os.system(f"dvc push -R {folder}")
