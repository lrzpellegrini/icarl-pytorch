from typing import List, Sequence, Union, Tuple, Any

import math
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader

from .training_utils import extract_features_from_layer

# This code was adapted from the one of lasagne.init
# which is distributed under MIT License (MIT)
# Original source: https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
class LasagneInitializer(object):
    def __call__(self, shape: Union[List[int], Tuple[int]]) -> Tensor:
        return self.sample(shape)

    def sample(self, shape: Union[List[int], Tuple[int]]) -> Tensor:
        raise NotImplementedError()


class LasagneNormal(LasagneInitializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return torch.normal(self.mean, self.std, size=shape)


class LasagneHe(LasagneInitializer):
    def __init__(self, initializer, gain: Any = 1.0, c01b: bool = False):
        if gain == 'relu':
            gain = math.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            fan_in = torch.prod(torch.tensor(shape[:3]))
        else:
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = torch.prod(torch.tensor(shape[1:]))
            else:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

        std = self.gain * math.sqrt(1.0 / fan_in)
        return self.initializer(std=std).sample(shape)


class LasagneHeNormal(LasagneHe):
    def __init__(self, gain=1.0, c01b=False):
        super().__init__(LasagneNormal, gain, c01b)
# End of lasagne-adapted init code


def make_theano_training_function(model: Module, criterion: Module, optimizer: Optimizer, x: Tensor, y: Tensor,
                                  device=None) -> \
        float:
    model.train()
    model.zero_grad()
    if device is not None:
        x = x.to(device)
        y = y.to(device)

    output = model(x)
    loss: Tensor = criterion(output, y)
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().item()


def make_theano_validation_function(model: Module, criterion: Module, feature_extraction_layer: str,
                                    x: Tensor, y: Tensor, device=None) -> (float, Tensor, Tensor):
    output: Tensor
    output_features: Tensor
    loss: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        output, output_features = extract_features_from_layer(model, feature_extraction_layer, x)
        loss = criterion(output, y)

    return loss.detach().cpu().item(), output, output_features


def make_theano_inference_function(model: Module, x: Tensor, device=None) -> Tensor:
    output: Tensor

    model.eval()
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
        output = model(x)

    return output


def make_theano_feature_extraction_function(model: Module, feature_extraction_layer: str, x: Tensor,
                                            device=None, **kwargs) -> Tensor:
    output_features: List[Tensor] = []

    x_dataset = TensorDataset(x)
    x_dataset_loader = DataLoader(x_dataset, **kwargs)

    model.eval()
    with torch.no_grad():
        for (patterns,) in x_dataset_loader:
            if device is not None:
                patterns = patterns.to(device)
            output_features.append(extract_features_from_layer(model, feature_extraction_layer, patterns)[1])

    return torch.cat(output_features)
