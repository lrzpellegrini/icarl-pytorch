from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module


def extract_features_from_layer(model: Module, layer_name: str, x: Tensor) -> (Tensor, Tensor):
    activation: Dict[str, Tensor] = {}

    def get_activation(name):
        def hook(model_hook: Module, x_hook: Tensor, out_hook: Tensor):
            activation[name] = out_hook.detach().cpu()
        return hook

    model.eval()
    with torch.no_grad():
        with getattr(model, layer_name).register_forward_hook(get_activation(layer_name)):
            output = model(x)

    return output, activation[layer_name]
