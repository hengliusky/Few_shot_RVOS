import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
from model1 import build_model
from datasets.sailvos import SAILVOSDataset
from datasets.gygo import GyGoVOSDataset
from tools.load_pretrained_weights import pre_trained_model_to_finetune
from datasets.refer_ytvos import YTVOSDataset
import opts
import tqdm
from datasets.sailvos_gao import build_sail_vos
from datasets.gygo_gao import build_gygo_vos

parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
args = parser.parse_args()
print(args.dataset_file)
args.masks = True
args.binary = True
args.with_box_refine = True
args.freeze_text_encoder = True
iterations = args.iterations_per_epoch
shots = args.shots
data_path = args.data_path
support_frames = args.support_frames
save_dir = 'output'
if args.dataset_file == 'sailvos':
    dataset_train, support_video_ids = build_sail_vos('train', data_path, support_frames=support_frames,
                                                      iterations=iterations, shots=shots)
    dataset_test, _ = build_sail_vos('test', data_path, support_list=support_video_ids)
    save_path_prefix = os.path.join(save_dir, 'sailvosf_gao_no')
else:
    dataset_train, support_video_ids = build_gygo_vos('train', data_path, support_frames=support_frames,
                                                      iterations=iterations, shots=shots)
    dataset_test, _ = build_gygo_vos('test', data_path, support_list=support_video_ids)
    save_path_prefix = os.path.join(save_dir, 'gygof_gao_no')
data_loader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, collate_fn=utils.collate_fn1)
data_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, collate_fn=utils.collate_fn3)

# test_list = dataset_test.get_class_list()
test_list = [1]
test_evaluations = Evaluation(class_list=test_list)
# model1, criterion, _ = build_model(args)
model1, criterion, _ = build_model(args)
device = args.device
model1.to(device)
# args.visualize = True
model1_without_ddp = model1
n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print('number of params:', n_parameters)
checkpoint = torch.load('pretrained_weights/ytvos_r50.pth', map_location='cpu')
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
print(len(data_loader_test))
args.query_frame = 20

for samples, atargets, idx, vid, frames_name in tqdm.tqdm(data_loader_test):
    samples = samples.to(device)
    captions = [t["caption"] for t in atargets]
    targets = utils.targets_to(atargets, device)

    bt, len_video, c, h, w = samples.tensors.shape
    step_len = (len_video // args.query_frame)
    if len_video % args.query_frame != 0:
        step_len = step_len + 1
    test_len = step_len
    for i in range(test_len):
        if i == step_len - 1:
            query_img = samples.tensors[:, i * args.query_frame:]
            query_mask = samples.mask[:, i * args.query_frame:]
            qsamples = utils.NestedTensor(query_img, query_mask)
            qtargets = {
                'labels': [t["labels"][i * args.query_frame:] for t in targets][0],  # [T,]
                'boxes': [t["boxes"][i * args.query_frame:] for t in targets][0],  # [T, 4], xyxy
                'masks': [t["masks"][i * args.query_frame:] for t in targets][0],  # [T, H, W]
                'valid': [t["valid"][i * args.query_frame:] for t in targets][0],  # [T,]
                'orig_size': [t["orig_size"] for t in targets][0],
                'size': [t["size"] for t in targets][0],
                # 'area': [t["area"][i * args.query_frame:] for t in targets]
            }
            mask = [t["masks"][i * args.query_frame:] for t in targets][0]
        else:
            query_img = samples.tensors[:, i * args.query_frame:(i + 1) * args.query_frame]
            query_mask = samples.mask[:, i * args.query_frame:(i + 1) * args.query_frame]
            qsamples = utils.NestedTensor(query_img, query_mask)
            qtargets = {
                'labels': [t["labels"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
                # [T,]
                'boxes': [t["boxes"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
                # [T, 4], xyxy
                'masks': [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
                # [T, H, W]
                'valid': [t["valid"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],  # [T,]
                'orig_size': [t["orig_size"] for t in targets][0],
                'size': [t["size"] for t in targets][0],
                # 'area': [t["area"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets]
            }
            mask = [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0]
        qqtargets = [qtargets]
        # qtargets = list(qtargets)
        with torch.no_grad():
            outputs = model1(qsamples, captions, qqtargets)
        # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
        pred_logits = outputs["pred_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_masks = outputs["pred_masks"][0]
        pred_ref_points = outputs["reference_points"][0]
        origin_h, origin_w = int(qtargets['size'][0]), int(qtargets['size'][1])
        len_frames, _, _ = mask.shape
        # len_frames, origin_h, origin_w = mask.shape
        # according to pred_logits, select the query index
        pred_scores = pred_logits.sigmoid()  # [t, q, k]
        pred_scores = pred_scores.mean(0)  # [q, k]
        max_scores, _ = pred_scores.max(-1)  # [q,]
        _, max_ind = max_scores.max(-1)  # [1,]
        max_inds = max_ind.repeat(len_frames)
        pred_masks = pred_masks[range(len_frames), max_inds, ...]  # [t, h, w]
        pred_masks = pred_masks.unsqueeze(0)  #
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask.float(), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        pred_masks = pred_masks.sigmoid()
        # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
        test_evaluations.update_evl(tuple([1]), mask, pred_masks)
        args.visualize = True
        if args.visualize:
            save_path = os.path.join(save_path_prefix, str(vid[0]))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(pred_masks.shape[1]):
                if i == step_len - 1:
                    sub_frames_name = frames_name[0][i * args.query_frame:]
                else:
                    sub_frames_name = frames_name[0][i * args.query_frame:(i + 1) * args.query_frame]
                frame_name = sub_frames_name[j]
                mask = pred_masks[0, j].cpu().numpy()
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, str(frame_name) + ".png")
                mask.save(save_file)

mean_f = np.mean(test_evaluations.f_score)
str_mean_f = 'F: %.4f ' % (mean_f)
mean_j = np.mean(test_evaluations.j_score)
str_mean_j = 'J: %.4f ' % (mean_j)

f_list = ['%.4f' % n for n in test_evaluations.f_score]
str_f_list = ' '.join(f_list)
j_list = ['%.4f' % n for n in test_evaluations.j_score]
str_j_list = ' '.join(j_list)

print(str_mean_f, str_f_list + '\n')
print(str_mean_j, str_j_list + '\n')


# ./train_sailvos.sh sailvos/r50 pretrained_weights/ytvos_r50_joint.pth --backbone resnet50
# ./train_gygo.sh gygo/r50 pretrained_weights/ytvos_r50_joint.pth --backbone resnet50
# ./train_sailvos.sh sailvos/r50 pretrained_weights/ytvos_r50.pth --backbone resnet50
# ./train_gygo.sh gygo/r50 pretrained_weights/ytvos_r50.pth --backbone resnet50
# ./train_ytvos.sh ytvos/r50  pretrained_weights/r50_pretrained.pth --backbone resnet50
# python test_nofinetune.py --dataset_file sailvos