from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import os

from ret_benchmark.config import cfg as g_cfg


def get_config_root_path():
    ''' Path to configs for unit tests '''
    # cur_file_dir is root/tests/env_tests
    cur_file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    ret = os.path.dirname(os.path.dirname(cur_file_dir))
    ret = os.path.join(ret, "configs")
    return ret


def load_config(rel_path):
    ''' Load config from file path specified as path relative to config_root '''
    cfg_path = os.path.join(get_config_root_path(), rel_path)
    return load_config_from_file(cfg_path)


def load_config_from_file(file_path):
    ''' Load config from file path specified as absolute path '''
    ret = copy.deepcopy(g_cfg)
    ret.merge_from_file(file_path)
    return ret
