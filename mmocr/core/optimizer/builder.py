import copy
import inspect

import torch

from mmcv.utils import Registry, build_from_cfg

from .ranger2020 import Ranger

from mmcv.runner.optimizer.builder import OPTIMIZERS
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            if module_name not in OPTIMIZERS.module_dict.keys():
                # default optimizer (eg. Adam, AdamW ...) has already register in mmcv.
                OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)

    # register Ranger optimizers
    if inspect.isclass(Ranger) and issubclass(Ranger,
                                              torch.optim.Optimizer):
        OPTIMIZERS.register_module()(Ranger)
        torch_optimizers.append(Ranger)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer