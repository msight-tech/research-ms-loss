# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
from collections import OrderedDict

import torch
from torch.nn.modules import Sequential

from .backbone import build_backbone
from .heads import build_head


def build_model(cfg):
    backbone = build_backbone(cfg)
    head = build_head(cfg)

    model = Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', head)
    ]))

    if cfg.MODEL.PRETRAIN == 'imagenet':
        print('Loading imagenet pretrianed model ...')
        pretrained_path = os.path.expanduser(cfg.MODEL.PRETRIANED_PATH[cfg.MODEL.BACKBONE.NAME])
        model.backbone.load_param(pretrained_path)
    elif os.path.exists(cfg.MODEL.PRETRAIN):
        ckp = torch.load(cfg.MODEL.PRETRAIN)
        model.load_state_dict(ckp['model'])
    return model
