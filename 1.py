import random

import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import argparse
import opts
from einops import rearrange, repeat
from pathlib import Path
import os
# parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
# args = parser.parse_args()
# # imgs = torch.FloatTensor(5,3,360,640)
# # imgs = [imgs]
# # if not isinstance(imgs, NestedTensor):
# #     samples = nested_tensor_from_videos_list(imgs)
# # print(samples)
# # group = 2
# # s = 'ytvos/r50-24/group_%d_nopre/checkpoint.pth'%(group)
# # print(s)
# a = torch.FloatTensor(5,128,40,80)
# b = repeat(a, 'b c h w -> (repeat b) c h w', repeat=2)
# print(b.shape)
# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt

# image1 原图
# # image2 分割图
# image1 = Image.open(r"D:\dataset\mini-SAIL-VOS\draw\image1.bmp")
# image2 = Image.open(r"D:\dataset\mini-SAIL-VOS\draw\image1.png")
#
# image1 = image1.convert('RGBA')
# image2 = image2.convert('RGBA')
#
# # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
# image = Image.blend(image1, image2, 0.3)
# image.save("test.png")
# image.show()
save_dir = 'results'
group = 2
save_path_prefix = os.path.join(save_dir, 'sailvosf', 'group_%d'%group)
print(save_path_prefix)