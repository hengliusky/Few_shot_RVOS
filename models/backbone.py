"""
Backbone modules.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from einops import rearrange
import opts
from util.misc import NestedTensor, is_main_process

from .position_encoding  import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    与batch normalization的工作原理类似，将统计量（均值与方差）和可学习的仿射参数固定住
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        # 需要将以上4个量注册到buffer，以便阻止梯度反向传播而更新它们，同时又能够记录在模型的state_dict中。
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:  # 需要记录每一层（ResNet的layer）的输出。
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"} deformable detr
            self.strides = [4, 8, 16, 32]   # 为了构建FPN结构
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # 获取网络中间几层的结果IntermediateLayerGetter（），它的输出是一个dict，对应了每层的输出，key是用户自定义的赋予输出特征的名字。 思想：先创建一个model ,
        # 然后把它传入IntermediateLayerGetter中，并传入一个字典，传入字典的key是model的直接的层，传入字典的value是返回字典中的key，返回字典的value对应的是model运行的中间结果。
        # 在定义它的时候注明作用的模型和要返回的layer，得到new_m。使用时喂输入变量，返回的就是对应的layer。

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将mask插值到与输出特征图尺寸一致
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)  # 将图像张量和对应的mask封装到一起。tensor就是输入的图像。mask跟tensor同高宽但是单通道
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        # self[0]是backbone，self[1]是position_embedding
        # Joiner 就是将backbone 和position encoding 集成到一个模块，使得前向过程中更方便地使用两者的功能
        # 前向过程就是对backbone 的每层输出都进行位置编码，最终返回backbone的输出及对应的位置编码结果。

    def forward(self, tensor_list: NestedTensor):
        temp_tensor_list = NestedTensor(rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w'), rearrange(tensor_list.mask, 'b t h w -> (b t) h w'))
        # temp_tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')
        # temp_tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')
        # tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')
        # tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')
        # b, t, _, _, _ = tensor_list.tensors.shape
        # xs = self[0](tensor_list)  # backbone的输出
        xs = self[0](temp_tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        # tensor_list.tensors = rearrange(tensor_list.tensors, '(b t) c h w ->b t c h w ', b=b, t =t)
        # tensor_list.mask = rearrange(tensor_list.mask, '(b t) c h w ->b t c h w ')
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels)  # 返回中间层 应该改为args.num_feature_levels，原为args.num
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


""" 以下为测试代码"""


# import argparse
# from datasets import build_dataset, get_coco_api_from_dataset
#
# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
#
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        nested_tensor_from_videos_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
#     args = parser.parse_args()
#     dataset_val = build_dataset('a2d', image_set='val', args=args)
#     """
#     backbone = build_backbone(args)
#     num_backbone_outs = len(backbone.strides[-3:])
#     print(num_backbone_outs)
#     input_proj_list = []
#     for _ in range(num_backbone_outs):
#         print(_)
#         in_channels = backbone.num_channels[-3:][_]
#         print(in_channels)
#
#     hidden_dim = 256
#     feature_channels = [backbone.num_channels[0]] + 3 * [hidden_dim]
#     print(feature_channels)
#     # Build Dynamic Conv
#     controller_layers =3  # 3
#     in_channels = 256  # 256
#     dynamic_mask_channels = 8  # 8
#     mask_out_stride = 4
#     mask_feat_stride = 4
#     rel_coord = False
#     weight_nums, bias_nums = [], []
#     num_queries = 5
#     random_refpoints_xy = True
#     num_patterns = 0
#     """
#     """for l in range(controller_layers):
#         if l == 0:
#             if rel_coord:
#                 weight_nums.append((in_channels + 2) * dynamic_mask_channels)
#             else:
#                 weight_nums.append(in_channels * dynamic_mask_channels)
#             bias_nums.append(dynamic_mask_channels)
#         elif l == controller_layers - 1:
#             weight_nums.append(dynamic_mask_channels * 1)  # output layer c -> 1
#             bias_nums.append(1)
#         else:
#             weight_nums.append(dynamic_mask_channels * dynamic_mask_channels)
#             bias_nums.append(dynamic_mask_channels)
#
#     weight_nums = weight_nums
#     bias_nums = bias_nums
#     num_gen_params = sum(weight_nums) + sum(bias_nums)
#     print(weight_nums)
#     print(bias_nums)
#     print(num_gen_params)
#     controller = MLP(hidden_dim, hidden_dim, num_gen_params, 3)
#     print(controller)
#     print("111")
#     for layer in controller.layers:
#         print("222")
#         print(controller.layers)
#         nn.init.zeros_(layer.bias)
#         print(layer.bias)
#         nn.init.xavier_uniform_(layer.weight)
#         print(layer.weight)"""
#     """
#     two_stage = False
#     use_dab = True
#     if not two_stage:
#         if not use_dab:
#             query_embed = nn.Embedding(num_queries, hidden_dim)  # 产生num_queries个长度为hid_dim的可学习编码向量
#             print("11")
#         else:
#             tgt_embed = nn.Embedding(num_queries, hidden_dim)
#             refpoint_embed = nn.Embedding(num_queries, 4)
#             if random_refpoints_xy:
#                 # import ipdb; ipdb.set_trace()
#                 refpoint_embed.weight.data[:, :2].uniform_(0, 1)
#                 refpoint_embed.weight.data[:, :2] = inverse_sigmoid(refpoint_embed.weight.data[:, :2])
#                 refpoint_embed.weight.data[:, :2].requires_grad = False
#             print(tgt_embed)
#             print(refpoint_embed)
#             print(refpoint_embed.weight.data)
#
#     if num_patterns > 0:
#         patterns_embed = nn.Embedding(num_patterns, hidden_dim)
#         print("222")
#         """
#
