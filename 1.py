import random

import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import argparse
import opts
from pathlib import Path
import os
parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
args = parser.parse_args()
# imgs = torch.FloatTensor(5,3,360,640)
# imgs = [imgs]
# if not isinstance(imgs, NestedTensor):
#     samples = nested_tensor_from_videos_list(imgs)
# print(samples)
# group = 2
# s = 'ytvos/r50-24/group_%d_nopre/checkpoint.pth'%(group)
# print(s)
f = str(0.456)
tmp_f =0.0
f = float(f)
a = f + tmp_f