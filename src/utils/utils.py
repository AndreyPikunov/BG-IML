from pathlib import Path

import yaml


def load_config(filename):
    with open(filename) as f:
        config = yaml.safe_load(f.read())
    return config


def get_abs_dirname(filename):
    result = Path(filename).parent.resolve()
    result = str(result)
    return result


def rescale(x, vmin=0, vmax=1):
    return (x - x.min()) / (x.max() - x.min()) * (vmax - vmin) + vmin
