# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        batch_idxs_dict, avai_labels = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch
