
"""
Training script of ReferFormer
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from util.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
import util.misc as utils
import datasets.samplers as samplers
from engine_referformer import train_one_epoch
from rvos_model import build_model
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import torch.nn.functional as F
import opts
import tqdm
from datasets.sailvos_gao import build_sail_vos



def main(args):
    args.masks = True
    args.distributed = False
    utils.init_distributed_mode(args)
    
    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessor = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names) 
                 and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
            "lr": args.lr_text_encoder,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)


    iterations = args.iterations_per_epoch
    shots = args.shots
    data_path = args.data_path
    support_frames = args.support_frames

    print(args.dataset_file)
    save_dir = 'output'
    dataset_train, support_video_ids = build_sail_vos('train', data_path, support_frames=support_frames, iterations=iterations, shots=shots)
    dataset_test, _ = build_sail_vos('test', data_path, support_list=support_video_ids)
    save_path_prefix = os.path.join(save_dir, 'sailvosf_gao')

    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)


    data_loader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, collate_fn=utils.collate_fn1)
    data_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, collate_fn=utils.collate_fn3)

    test_list = [1]
    test_evaluations = Evaluation(class_list=test_list)

    if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
        print("============================================>")
        print("Load pretrained weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



    model1, criterion, _ = build_model(args)
    device = args.device
    model1.to(device)

    model1_without_ddp = model1
    n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.dataset_file == 'sailvos':
        checkpoint = torch.load('sailvos/r50_gao/checkpoint.pth', map_location='cpu')
    else:
        checkpoint = torch.load('gygo/r50_gao/checkpoint.pth', map_location='cpu')
    
    missing_keys, unexpected_keys = model1_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    

    model1.eval()
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
                    'labels': [t["labels"][i * args.query_frame:] for t in targets][0],
                    'boxes': [t["boxes"][i * args.query_frame:] for t in targets][0],
                    'masks': [t["masks"][i * args.query_frame:] for t in targets][0],
                    'valid': [t["valid"][i * args.query_frame:] for t in targets][0],
                    'orig_size': [t["orig_size"] for t in targets][0],
                    'size': [t["size"] for t in targets][0],

                }
                mask = [t["masks"][i * args.query_frame:] for t in targets][0]
            else:
                query_img = samples.tensors[:, i * args.query_frame:(i + 1) * args.query_frame]
                query_mask = samples.mask[:, i * args.query_frame:(i + 1) * args.query_frame]
                qsamples = utils.NestedTensor(query_img, query_mask)
                qtargets = {
                    'labels': [t["labels"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],

                    'boxes': [t["boxes"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],

                    'masks': [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],

                    'valid': [t["valid"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0],
                    'orig_size': [t["orig_size"] for t in targets][0],
                    'size': [t["size"] for t in targets][0],

                }
                mask = [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in targets][0]
            qqtargets = [qtargets]

            with torch.no_grad():
                outputs = model1(qsamples, captions, qqtargets)

            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_ref_points = outputs["reference_points"][0]
            origin_h, origin_w = int(qtargets['size'][0]), int(qtargets['size'][1])
            len_frames, _, _ = mask.shape


            pred_scores = pred_logits.sigmoid()
            pred_scores = pred_scores.mean(0)
            max_scores, _ = pred_scores.max(-1)
            _, max_ind = max_scores.max(-1)
            max_inds = max_ind.repeat(len_frames)
            pred_masks = pred_masks[range(len_frames), max_inds, ...]
            pred_masks = pred_masks.unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask = F.interpolate(mask.float(), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = pred_masks.sigmoid()

            test_evaluations.update_evl(tuple([1]), mask, pred_masks)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





