from dvc.api import read
import subprocess


def pull_data():
    subprocess.run(["dvc", "pull"], check=True)