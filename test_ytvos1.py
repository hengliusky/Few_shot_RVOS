"""

Training script of ReferFormer
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import xlwt, xlrd
import xlutils.copy

from util.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
from model1 import build_model, few_build_model
from datasets.sailvos import SAILVOSDataset
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import torch.nn.functional as F
import opts
from datasets.gygo import GyGoVOSDataset
from datasets.refer_ytvos import YTVOSDataset
from datasets.sailvos_2exp import build_sail_vos
from datasets.gygo_2exp import build_gygo_vos
from datasets.ytvos_2exp import build_yt_vos

def main(args):
    args.masks = True
    # args.dataset_file = 'mini-ytvos'
    args.binary = True
    args.with_box_refine = True
    args.freeze_text_encoder = True
    utils.init_distributed_mode(args)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # args.distributed = False
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessor = few_build_model(args)
    # model.to(device)
    #
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    #
    # # for n, p in model_without_ddp.named_parameters():
    # #     print(n)
    #
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    #
    # def match_name_keywords(n, name_keywords):
    #     out = False
    #     for b in name_keywords:
    #         if b in n:
    #             out = True
    #             break
    #     return out
    #
    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
    #                                                                                                args.lr_text_encoder_names)
    #              and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if
    #                    match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if
    #                    match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
    #         "lr": args.lr_text_encoder,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if
    #                    match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr * args.lr_linear_proj_mult,
    #     }
    # ]
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    print(args.dataset_file)
    save_dir = 'output'
    # if args.dataset_file == 'sailvos':
    #     dataset_test = SAILVOSDataset(train=False, query_frame=5, support_frame=5, cat_id=args.cat_id)
    #     save_path_prefix = os.path.join(save_dir, 'sailvosf')
    # elif args.dataset_file == 'mini-ytvos':
    #     dataset_test = YTVOSDataset( train=False, query_frame=5,support_frame=5, set_index=args.group)
    #     save_path_prefix = os.path.join(save_dir, 'ytvosf')
    # else:
    #     dataset_test = GyGoVOSDataset(train=False, query_frame=5, support_frame=5, cat_id=args.cat_id)
    #     save_path_prefix = os.path.join(save_dir, 'gygof')
    # if not os.path.exists(save_path_prefix):
    #     os.makedirs(save_path_prefix)

    # data_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)

    data_path = args.data_path
    if args.dataset_file == 'sailvos':
        dataset_test, _ = build_sail_vos('test', data_path)
        save_path_prefix = os.path.join(save_dir, 'sailvosf')
    elif args.dataset_file == 'gygo':
        dataset_test, _ = build_gygo_vos('test', data_path)
        save_path_prefix = os.path.join(save_dir, 'gygof')
    else:
        dataset_test = build_yt_vos('test', data_path, set_index=args.group, support_frames=args.support_frames)
        save_path_prefix = os.path.join(save_dir, 'ytvosf')

    data_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)


    test_list = dataset_test.get_class_list()
    test_evaluations = Evaluation(class_list=test_list)

    model1, criterion, _ = few_build_model(args)
    device = args.device
    model1.to(device)

    model1_without_ddp = model1
    n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoint = torch.load('ytvos/r50_base_24/checkpoint.pth', map_location='cpu')
    missing_keys, unexpected_keys = model1_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    # start inference
    model1.eval()
    print(len(data_loader_test))
    args.query_frame = 20
    samples, targets = None, None
    workbook = xlwt.Workbook(encoding='utf-8')
    ws = workbook.add_sheet('f')
    workbook.save('f_results_base.xls')
    for  index, data in tqdm(enumerate(data_loader_test), total=len(data_loader_test), leave=True):
        qsamples, qtargets, new_samples, new_targets, idx, vid, begin_new, frames_name, obj_id, exp_id = data
        if begin_new[0]:
            samples, targets = new_samples, new_targets
        ssamples = samples.to(device)
        scaptions = [t["caption"] for t in targets]
        stargets = utils.targets_to(targets, device)


        qsamples = qsamples.to(device)
        q_captions = [t["caption"] for t in qtargets]
        qtargets = utils.targets_to(qtargets, device)

        bt, len_video, c, h, w = qsamples.tensors.shape
        step_len = (len_video // args.query_frame)
        if len_video % args.query_frame != 0:
            step_len = step_len + 1
        test_len = step_len
        tmp_f = 0
        tmp_j = 0
        filename = "f_results_base.xls"
        for i in range(test_len):
            if i == step_len - 1:
                query_img = qsamples.tensors[:, i * args.query_frame:]
                query_mask = qsamples.mask[:, i * args.query_frame:]
                q_samples = utils.NestedTensor(query_img, query_mask)
                q_targets = {
                    'labels': [t["labels"][i * args.query_frame:] for t in qtargets][0],  # [T,]
                    'boxes': [t["boxes"][i * args.query_frame:] for t in qtargets][0],  # [T, 4], xyxy
                    'masks': [t["masks"][i * args.query_frame:] for t in qtargets][0],  # [T, H, W]
                    'valid': [t["valid"][i * args.query_frame:] for t in qtargets][0],  # [T,]
                    'orig_size': [t["orig_size"] for t in qtargets][0],
                    'size': [t["size"] for t in qtargets][0],
                    # 'area': [t["area"][i * args.query_frame:] for t in targets]
                }
                mask = [t["masks"][i * args.query_frame:] for t in qtargets][0]
            else:
                query_img = qsamples.tensors[:, i * args.query_frame:(i + 1) * args.query_frame]
                query_mask = qsamples.mask[:, i * args.query_frame:(i + 1) * args.query_frame]
                q_samples = utils.NestedTensor(query_img, query_mask)
                q_targets = {
                    'labels': [t["labels"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],
                    # [T,]
                    'boxes': [t["boxes"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],
                    # [T, 4], xyxy
                    'masks': [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],
                    # [T, H, W]
                    'valid': [t["valid"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],  # [T,]
                    'orig_size': [t["orig_size"] for t in qtargets][0],
                    'size': [t["size"] for t in qtargets][0],
                    # 'area': [t["area"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets]
                }
                mask = [t['masks'][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0]
            qqtargets = [q_targets]
            # qtargets = list(qtargets)

            s_samples = ssamples
            s_captions = scaptions
            s_targets = stargets
            with torch.no_grad():
                outputs = model1(q_samples, q_captions, qqtargets, s_samples, s_captions, s_targets)
            # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_ref_points = outputs["reference_points"][0]
            origin_h, origin_w = int(q_targets['size'][0]), int(q_targets['size'][1])
            len_frames, _, _ = mask.shape
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
            pred_masks = torch.nn.Sigmoid()(pred_masks)
            # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
            test_evaluations.update_evl(idx, mask, pred_masks)
            tmp_f += test_evaluations.tmp_f
            tmp_j += test_evaluations.tmp_j
            if args.visualize:
                save_path = os.path.join(save_path_prefix, str(vid[0]), str(exp_id[0]))
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
        f = tmp_f / len_video
        j = tmp_j / len_video
        write(filename, vid, obj_id, exp_id, j, f)
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


def readline(filename):
    wb = xlrd.open_workbook(filename,formatting_info=True)  #打开excel，保留文件格式
    sheet1 = wb.sheet_by_index(0)  #获取第一张表
    nrows = sheet1.nrows  #获取总行数
    # ncols = sheet1.ncols
    return nrows
def write(filename, vid, obj_id, exp_id, j, f):
    data = xlrd.open_workbook(filename)
    ws = xlutils.copy.copy(data) #复制之前表里存在的数据
    table = ws.get_sheet(0)
    nownrows = readline(filename)
    table.write(nownrows, 0, label=str(vid[0]))  #最后一行追加数据
    table.write(nownrows, 1, label=str(obj_id[0]))
    table.write(nownrows, 2, label=str(exp_id[0]))
    table.write(nownrows, 3, label=str(j))
    table.write(nownrows, 4, label=str(f))
    if f<0.3:
        table.write(nownrows, 5, label='True')
    ws.save(filename)  #保存的有旧数据和新数据



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



