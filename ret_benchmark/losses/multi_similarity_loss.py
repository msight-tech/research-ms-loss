# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ret_benchmark.losses.registry import LOSS


@LOSS.register('ms_loss')
class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss
