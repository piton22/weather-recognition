import subprocess


def pull_data():
    subprocess.run(["dvc", "pull"], check=True)
