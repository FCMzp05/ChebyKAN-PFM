import os


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
