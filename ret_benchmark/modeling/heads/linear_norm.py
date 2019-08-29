# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ret_benchmark.modeling.registry import HEADS
from ret_benchmark.utils.init_methods import weights_init_kaiming


@HEADS.register('linear_norm')
class LinearNorm(nn.Module):
    def __init__(self, cfg, in_channels):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, cfg.MODEL.HEAD.DIM)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
