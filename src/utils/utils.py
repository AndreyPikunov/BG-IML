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


def load_resnet(model_name):

    if model_name == "resnet18":
        from torchvision.models import (
            resnet18 as resnet,
            ResNet18_Weights as resnet_weights,
        )
    elif model_name == "resnet34":
        from torchvision.models import (
            resnet34 as resnet,
            ResNet34_Weights as resnet_weights,
        )
    elif model_name == "resnet50":
        from torchvision.models import (
            resnet50 as resnet,
            ResNet50_Weights as resnet_weights,
        )
    elif model_name == "resnet101":
        from torchvision.models import (
            resnet101 as resnet,
            ResNet101_Weights as resnet_weights,
        )
    else:
        raise RuntimeError()

    return resnet, resnet_weights
