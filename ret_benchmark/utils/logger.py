# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import sys
import logging

_streams = {
    "stdout": sys.stdout
}


def setup_logger(name: str, level: int, stream: str = "stdout") -> logging.Logger:
    global _streams
    if stream not in _streams:
        log_folder = os.path.dirname(stream)
        os.makedirs(log_folder, exist_ok=True)
        _streams[stream] = open(stream, 'w')
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    sh = logging.StreamHandler(stream=_streams[stream])
    sh.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
