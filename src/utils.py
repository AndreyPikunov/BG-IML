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
