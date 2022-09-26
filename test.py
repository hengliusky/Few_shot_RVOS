import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from util.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
from models import build_model, few_build_model
from datasets.sailvos import SAILVOSDataset
from datasets.gygo import GyGoVOSDataset
from tools.load_pretrained_weights import pre_trained_model_to_finetune
from datasets.refer_ytvos import YTVOSDataset
import opts
from tqdm import tqdm

parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
args = parser.parse_args()
print(args.dataset_file)
args.dataset_file = 'gygo'
args.binary = True
output_dir = 'output'
args.cat_id = 3
if args.dataset_file == 'sailvos':
    dataset_train = SAILVOSDataset(train=False, query_frame=5, support_frame=5, cat_id=args.cat_id)
    save_path_prefix = os.path.join(output_dir, 'sailvosf')
else:
    dataset_train = GyGoVOSDataset(train=False, query_frame=5, support_frame=5, cat_id=args.cat_id)
    save_path_prefix = os.path.join(output_dir, 'gygo')
# dataset_train = YTVOSDataset(query_frame=5, support_frame=5,
#                                 sample_per_class=args.sample_per_class,
#                                 set_index=args.group)
# dataset_train = SAILVOSDataset(train=True, query_frame=5, support_frame=5, cat_id=args.cat_id)
# sampler_train = torch.utils.data.RandomSampler(dataset_train)
# batch_sampler_train = torch.utils.data.BatchSampler(
#     sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)
# data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
#                                collate_fn=utils.collate_fn, num_workers=0)
test_list = dataset_train.get_class_list()
test_evaluations = Evaluation(class_list=test_list)
# model1, criterion, _ = build_model(args)
model1, criterion, _ = few_build_model(args)
device = args.device
model1.to(device)
args.visualize = True
model1_without_ddp = model1
n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# checkpoint = torch.load('ytvos_dirs/r50/checkpoint.pth', map_location='cpu')
# checkpoint = torch.load('pretrained_weights/ytvos_r50_joint.pth', map_location='cpu')
checkpoint = torch.load('sailvos/r50/checkpoint.pth', map_location='cpu')
missing_keys, unexpected_keys = model1_without_ddp.load_state_dict(checkpoint['model'], strict=False)
unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
if len(missing_keys) > 0:
    print('Missing Keys: {}'.format(missing_keys))
if len(unexpected_keys) > 0:
    print('Unexpected Keys: {}'.format(unexpected_keys))
# # start inference
model1.eval()
# metric_logger = utils.MetricLogger(delimiter="  ")
# metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
# print_freq = 10
# for _, _, samples, targets in metric_logger.log_every(data_loader_train, print_freq):
print(len(data_loader_train))
args.query_frame = 25

for index, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True):
    qsamples, qtargets,ssamples, stargets, idx, vid, begin_new, frames_name = data
    ssamples = ssamples.to(device)
    scaptions = [t["caption"] for t in stargets]
    stargets = utils.targets_to(stargets, device)
    # outputs = model1(ssamples, scaptions, stargets)
    qsamples = qsamples.to(device)
    q_captions = [t["caption"] for t in qtargets]
    qtargets = utils.targets_to(qtargets, device)
    outputs = model1(qsamples, q_captions, qtargets, ssamples, scaptions, stargets)

    # if begin_new[0]:
    #     outputs = model(ssamples, captions, stargets)
    # else:
    #     continue
#     loss_dict = criterion(outputs, qtargets)
#
#     weight_dict = criterion.weight_dict
#     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#
#     # reduce losses over all GPUs for logging purposes
#     loss_dict_reduced = utils.reduce_dict(loss_dict)
#     loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                   for k, v in loss_dict_reduced.items()}
#     loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                 for k, v in loss_dict_reduced.items() if k in weight_dict}
#     losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#
#     loss_value = losses_reduced_scaled.item()
#
# print("test")
# for new_samples, new_targets, qsamples, qtargets, idx, begin_new, vid, frames_name in data_loader_train:
#     if begin_new[0]:
#         samples, targets = new_samples, new_targets
    # if not math.isfinite(loss_value):
    #     print("Loss is {}, stopping training".format(loss_value))
    #     print(loss_dict_reduced)
    #     sys.exit(1)
    # optimizer.zero_grad()
    # losses.backward()
    # if max_norm > 0:
    #     grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # else:
    #     grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
    # optimizer.step()
    #
    # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
    # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # metric_logger.update(grad_norm=grad_total_norm)
#     samples = samples.to(device)
#     captions = [t["caption"] for t in atargets]
#     targets = utils.targets_to(atargets, device)
# #     with torch.no_grad():
# #         outputs = model1(samples, captions, targets)
# #     # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
# #     pred_logits = outputs["pred_logits"][0]
# #     pred_boxes = outputs["pred_boxes"][0]
# #     pred_masks = outputs["pred_masks"][0]
# #     pred_ref_points = outputs["reference_points"][0]
# #     mask = [t["masks"] for t in targets][0]
# #     len_frames, origin_h, origin_w = mask.shape
# #
# #     # according to pred_logits, select the query index
# #     pred_scores = pred_logits.sigmoid()  # [t, q, k]
# #     pred_scores = pred_scores.mean(0)  # [q, k]
# #     max_scores, _ = pred_scores.max(-1)  # [q,]
# #     _, max_ind = max_scores.max(-1)  # [1,]
# #     max_inds = max_ind.repeat(len_frames)
# #     pred_masks = pred_masks[range(len_frames), max_inds, ...]  # [t, h, w]
# #     pred_masks = pred_masks.unsqueeze(0)  #
# #     mask = mask.unsqueeze(0)
# #     pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
# #     # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
# #     test_evaluations.update_evl(idx, mask, pred_masks)
# #     if args.visualize:
# #         save_path = os.path.join(save_path_prefix, str(vid[0]))
# #         if not os.path.exists(save_path):
# #             os.makedirs(save_path)
# #         for j in range(pred_masks.shape[1]):
# #             frame_name = frames_name[0][j]
# #             mask = pred_masks[0, j].cpu().numpy()
# #             mask[mask < 0.5] = 0
# #             mask[mask >= 0.5] = 1
# #             mask = Image.fromarray(mask * 255).convert('L')
# #             save_file = os.path.join(save_path, str(frame_name) + ".png")
# #             mask.save(save_file)
# # mean_f = np.mean(test_evaluations.f_score)
# # str_mean_f = 'F: %.4f ' % (mean_f)
# # mean_j = np.mean(test_evaluations.j_score)
# # str_mean_j = 'J: %.4f ' % (mean_j)
# #
# # f_list = ['%.4f' % n for n in test_evaluations.f_score]
# # str_f_list = ' '.join(f_list)
# # j_list = ['%.4f' % n for n in test_evaluations.j_score]
# # str_j_list = ' '.join(j_list)
# # print(str_mean_f, str_f_list + '\n')
# # print(str_mean_j, str_j_list + '\n')
#
#     bt, len_video, c, h, w = samples.tensors.shape
#     step_len = (len_video // args.query_frame)
#     if len_video % args.query_frame != 0:
#         step_len = step_len + 1
#     test_len = step_len
#     for i in range(test_len):
#         if i == step_len - 1:
#             query_img = samples.tensors[:, i * args.query_frame:]
#             query_mask = samples.mask[:, i * args.query_frame:]
#             qsamples = utils.NestedTensor(query_img, query_mask)
#             qtargets = {
#                 'labels': [t["labels"][i * args.query_frame:] for t in targets][0],  # [T,]
#                 'boxes': [t["boxes"][i * args.query_frame:] for t in targets][0],  # [T, 4], xyxy
#                 'masks': [t["masks"][i * args.query_frame:] for t in targets][0],  # [T, H, W]
#                 'valid': [t["valid"][i * args.query_frame:] for t in targets][0],  # [T,]
#                 'orig_size': [t["orig_size"] for t in targets][0],
#                 'size': [t["size"] for t in targets][0],
#                 # 'area': [t["area"][i * args.query_frame:] for t in targets]
#             }
#             mask = [t["masks"][i * args.query_frame:] for t in targets][0]
#         else:
#             query_img = samples.tensors[:, i * args.query_frame:(i + 1) * args.query_frame]
#             query_mask = samples.mask[:, i * args.query_frame:(i + 1) * args.query_frame]
#             qsamples = utils.NestedTensor(query_img, query_mask)
#             qtargets = {
#                 'labels': [t["labels"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
#                 # [T,]
#                 'boxes': [t["boxes"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
#                 # [T, 4], xyxy
#                 'masks': [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
#                 # [T, H, W]
#                 'valid': [t["valid"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],  # [T,]
#                 'orig_size': [t["orig_size"] for t in targets][0],
#                 'size': [t["size"] for t in targets][0],
#                 # 'area': [t["area"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets]
#             }
#             mask = [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0]
#         qqtargets = [qtargets]
#         # qtargets = list(qtargets)
#         with torch.no_grad():
#             outputs = model1(qsamples, captions, qqtargets)
#         # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
#         pred_logits = outputs["pred_logits"][0]
#         pred_boxes = outputs["pred_boxes"][0]
#         pred_masks = outputs["pred_masks"][0]
#         pred_ref_points = outputs["reference_points"][0]
#         len_frames, origin_h, origin_w = mask.shape
#         # according to pred_logits, select the query index
#         pred_scores = pred_logits.sigmoid()  # [t, q, k]
#         pred_scores = pred_scores.mean(0)  # [q, k]
#         max_scores, _ = pred_scores.max(-1)  # [q,]
#         _, max_ind = max_scores.max(-1)  # [1,]
#         max_inds = max_ind.repeat(len_frames)
#         pred_masks = pred_masks[range(len_frames), max_inds, ...]  # [t, h, w]
#         pred_masks = pred_masks.unsqueeze(0)  #
#         mask = mask.unsqueeze(0)
#         pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
#         # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
#         test_evaluations.update_evl(idx, mask, pred_masks)
#         if args.visualize:
#             save_path = os.path.join(save_path_prefix, str(vid[0]))
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             for j in range(pred_masks.shape[1]):
#                 if i == step_len - 1:
#                     sub_frames_name = frames_name[0][i * args.query_frame:]
#                 else:
#                     sub_frames_name = frames_name[0][i * args.query_frame:(i + 1) * args.query_frame]
#                 frame_name = sub_frames_name[j]
#                 mask = pred_masks[0, j].cpu().numpy()
#                 mask[mask < 0.5] = 0
#                 mask[mask >= 0.5] = 1
#                 mask = Image.fromarray(mask * 255).convert('L')
#                 save_file = os.path.join(save_path, str(frame_name) + ".png")
#                 mask.save(save_file)
# mean_f = np.mean(test_evaluations.f_score)
# str_mean_f = 'F: %.4f ' % (mean_f)
# mean_j = np.mean(test_evaluations.j_score)
# str_mean_j = 'J: %.4f ' % (mean_j)
#
# f_list = ['%.4f' % n for n in test_evaluations.f_score]
# str_f_list = ' '.join(f_list)
# j_list = ['%.4f' % n for n in test_evaluations.j_score]
# str_j_list = ' '.join(j_list)
#
# print(str_mean_f, str_f_list + '\n')
# print(str_mean_j, str_j_list + '\n')


# ./train_sailvos.sh sailvos/r50 pretrained_weights/ytvos_r50_joint.pth --backbone resnet50
# ./train_gygo.sh gygo/r50 pretrained_weights/ytvos_r50_joint.pth --backbone resnet50
# ./train_sailvos.sh sailvos/r50 pretrained_weights/ytvos_r50.pth --backbone resnet50
# ./train_gygo.sh gygo/r50 pretrained_weights/ytvos_r50.pth --backbone resnet50
# ./train_ytvos.sh ytvos/r50  pretrained_weights/r50_pretrained.pth --backbone resnet50