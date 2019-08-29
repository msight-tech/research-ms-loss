# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension


requirements = ["torch", "torchvision"]

setup(
    name="ret_benchmark",
    version="0.1",
    author="Malong Technologies",
    url="https://github.com/MalongTech/research-ms-loss",
    description="ms-loss",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=requirements,
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
