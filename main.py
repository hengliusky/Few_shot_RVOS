"""

Training script of ReferFormer
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate, evaluate_a2d
# from models import build_model
from models import build_model
from datasets.sailvos import SAILVOSDataset
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import torch.nn.functional as F
import opts
from datasets.gygo import GyGoVOSDataset
from tqdm import tqdm

def main(args):
    args.masks = True

    utils.init_distributed_mode(args)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.distributed = False
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

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

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

    # no validation ground truth for ytvos dataset
    # dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
    # dataset_train = SAILVOSDataset(train=False, query_frame=5, support_frame=5)
    print(args.dataset_file)
    save_dir = 'output'
    if args.dataset_file == 'sailvos':
        dataset_train = SAILVOSDataset(train=True, query_frame=5, support_frame=5, cat_id=args.cat_id)
        save_path_prefix = os.path.join(save_dir, 'sailvosf')
    else:
        dataset_train = GyGoVOSDataset(train=False, query_frame=5, support_frame=5, cat_id=args.cat_id)
        save_path_prefix = os.path.join(save_dir, 'gygof')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    test_list = dataset_train.get_class_list()
    test_evaluations = Evaluation(class_list=test_list)
    # A2D-Sentences
    if args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
        dataset_val = build_dataset(args.dataset_file, image_set='val', args=args)
        if args.distributed:
            if args.cache_mode:
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    
    if args.dataset_file == "davis":
        assert args.pretrained_weights is not None, "Please provide the pretrained weight to finetune for Ref-DAVIS17"
        print("============================================>")
        print("Ref-DAVIS17 are finetuned using the checkpoint trained on Ref-Youtube-VOS")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")

    if args.dataset_file == "jhmdb":
        assert args.resume is not None, "Please provide the checkpoint to resume for JHMDB-Sentences"
        print("============================================>")
        print("JHMDB-Sentences are directly evaluated using the checkpoint trained on A2D-Sentences")
        print("Load checkpoint weights from {} ...".format(args.pretrained_weights))
        # load checkpoint in the args.resume
        print("============================================>")

    # for Ref-Youtube-VOS and A2D-Sentences
    # finetune using the pretrained weights on Ref-COCO
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
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
                    'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
        test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
        return


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
            # extra checkpoint before LR drop and every epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
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

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.dataset_file == 'a2d':
            test_stats = evaluate_a2d(model, data_loader_val, postprocessor, device, args)
            log_stats.update({**{f'{k}': v for k, v in test_stats.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    ###### test
    # model
    model1, criterion, _ = build_model(args)
    device = args.device
    model1.to(device)

    model1_without_ddp = model1
    n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.dataset_file == 'sailvos':
        checkpoint = torch.load('sailvos/r50/checkpoint.pth', map_location='cpu')
    else:
        checkpoint = torch.load('gygo/r50/checkpoint.pth', map_location='cpu')
    missing_keys, unexpected_keys = model1_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    # start inference
    model1.eval()
    print(len(data_loader_train))
    args.query_frame = 15
    # for a, b, samples, atargets, idx, begin_new, vid, frames_name in data_loader_train:
    #     samples = samples.to(device)
    #     captions = [t["caption"] for t in atargets]
    #     targets = utils.targets_to(atargets, device)
    #     with torch.no_grad():
    #         outputs = model1(samples, captions, targets)
    #     # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
    #     pred_logits = outputs["pred_logits"][0]
    #     pred_masks = outputs["pred_masks"][0]
    #     mask = [t["masks"] for t in targets][0]
    #     len_frames, origin_h, origin_w = mask.shape
    #
    #     # according to pred_logits, select the query index
    #     pred_scores = pred_logits.sigmoid()  # [t, q, k]
    #     pred_scores = pred_scores.mean(0)  # [q, k]
    #     max_scores, _ = pred_scores.max(-1)  # [q,]
    #     _, max_ind = max_scores.max(-1)  # [1,]
    #     max_inds = max_ind.repeat(len_frames)
    #     pred_masks = pred_masks[range(len_frames), max_inds, ...]  # [t, h, w]
    #     pred_masks = pred_masks.unsqueeze(0)  #
    #     mask = mask.unsqueeze(0)
    #     pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
    #     # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
    #     test_evaluations.update_evl(idx, mask, pred_masks)
    #     if args.visualize:
    #         save_path = os.path.join(save_path_prefix, str(vid[0]))
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         for j in range(pred_masks.shape[1]):
    #             frame_name = frames_name[0][j]
    #             mask = pred_masks[0, j].cpu().numpy()
    #             mask[mask < 0.5] = 0
    #             mask[mask >= 0.5] = 1
    #             mask = Image.fromarray(mask * 255).convert('L')
    #             save_file = os.path.join(save_path, str(frame_name) + ".png")
    #             mask.save(save_file)
    # mean_f = np.mean(test_evaluations.f_score)
    # str_mean_f = 'F: %.4f ' % (mean_f)
    # mean_j = np.mean(test_evaluations.j_score)
    # str_mean_j = 'J: %.4f ' % (mean_j)
    #
    # f_list = ['%.4f' % n for n in test_evaluations.f_score]
    # str_f_list = ' '.join(f_list)
    # j_list = ['%.4f' % n for n in test_evaluations.j_score]
    # str_j_list = ' '.join(j_list)
    # print(str_mean_f, str_f_list + '\n')
    # print(str_mean_j, str_j_list + '\n')
    for index, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), leave=True):
        qsamples, qtargets, new_samples, new_targets, idx, vid, begin_new, frames_name = data
        qsamples = qsamples.to(device)
        q_captions = [t["caption"] for t in qtargets]
        qtargets = utils.targets_to(qtargets, device)

        bt, len_video, c, h, w = qsamples.tensors.shape
        step_len = (len_video // args.query_frame)
        if len_video % args.query_frame != 0:
            step_len = step_len + 1
        test_len = step_len
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

            with torch.no_grad():
                outputs = model1(q_samples, q_captions, qqtargets)
            # pred_masks = outputs["pred_masks"][0]  # F, 5 ,h,w
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_ref_points = outputs["reference_points"][0]
            # origin_h, origin_w = int(q_targets['orig_size'][0]), int(q_targets['orig_size'][1])
            len_frames, origin_h, origin_w = mask.shape
            # according to pred_logits, select the query index
            pred_scores = pred_logits.sigmoid()  # [t, q, k]
            pred_scores = pred_scores.mean(0)  # [q, k]
            max_scores, _ = pred_scores.max(-1)  # [q,]
            _, max_ind = max_scores.max(-1)  # [1,]
            max_inds = max_ind.repeat(len_frames)
            pred_masks = pred_masks[range(len_frames), max_inds, ...]  # [t, h, w]
            pred_masks = pred_masks.unsqueeze(0)  #
            mask = mask.unsqueeze(0)
            # mask = F.interpolate(mask.float(), size=(origin_h, origin_w), mode='bilinear', align_corners=True)
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=True)
            pred_masks = torch.nn.Sigmoid()(pred_masks)
            # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()
            test_evaluations.update_evl(idx, mask, pred_masks)
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



