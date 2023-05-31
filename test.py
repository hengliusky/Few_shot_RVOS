
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
from models import few_build_model
import torch.nn.functional as F
import opts
from datasets.ref_sailvos import build_sail_vos

from datasets.ref_ytvos import build_yt_vos
import matplotlib.pyplot as plt
def main(args, rand):
    seed = args.seed+rand
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.masks = True

    args.binary = True
    args.with_box_refine = True
    args.freeze_text_encoder = True

    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    print(args.dataset_file)
    save_dir = 'results'

    data_path = args.data_path
    if args.dataset_file == 'sailvos':
        dataset_test, _ = build_sail_vos('test', data_path)
        save_path_prefix = os.path.join(save_dir, 'sailvosf', 'group_%d'%args.group)

    else:
        dataset_test = build_yt_vos('test', data_path, set_index=args.group, support_frames=args.support_frames)
        save_path_prefix = os.path.join(save_dir, 'ytvosf', 'group_%d'%args.group)

    data_loader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, collate_fn=utils.collate_fn2)


    test_list = [1]
    test_evaluations = Evaluation(class_list=test_list)

    model1, criterion, _ = few_build_model(args)
    device = args.device
    model1.to(device)


    model1_without_ddp = model1
    n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    model_path = 'ytvos/r50-24-exp45/group_%d/checkpoint.pth'%(args.group)
    print(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')


    missing_keys, unexpected_keys = model1_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    model1.eval()
    print(len(data_loader_test))
    args.query_frame = 10
    samples, targets = None, None
    args.visualize = True
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
        filename = "f_results.xls"
        for i in range(test_len):
            if i == step_len - 1:
                query_img = qsamples.tensors[:, i * args.query_frame:]
                query_mask = qsamples.mask[:, i * args.query_frame:]
                q_samples = utils.NestedTensor(query_img, query_mask)
                q_targets = {
                    'labels': [t["labels"][i * args.query_frame:] for t in qtargets][0],
                    'boxes': [t["boxes"][i * args.query_frame:] for t in qtargets][0],
                    'masks': [t["masks"][i * args.query_frame:] for t in qtargets][0],
                    'valid': [t["valid"][i * args.query_frame:] for t in qtargets][0],
                    'orig_size': [t["orig_size"] for t in qtargets][0],
                    'size': [t["size"] for t in qtargets][0],

                }
                mask = [t["masks"][i * args.query_frame:] for t in qtargets][0]
            else:
                query_img = qsamples.tensors[:, i * args.query_frame:(i + 1) * args.query_frame]
                query_mask = qsamples.mask[:, i * args.query_frame:(i + 1) * args.query_frame]
                q_samples = utils.NestedTensor(query_img, query_mask)
                q_targets = {
                    'labels': [t["labels"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],

                    'boxes': [t["boxes"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],

                    'masks': [t["masks"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],

                    'valid': [t["valid"][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0],
                    'orig_size': [t["orig_size"] for t in qtargets][0],
                    'size': [t["size"] for t in qtargets][0],

                }
                mask = [t['masks'][i * args.query_frame:(i + 1) * args.query_frame] for t in qtargets][0]
            qqtargets = [q_targets]


            s_samples = ssamples
            s_captions = scaptions
            s_targets = stargets
            with torch.no_grad():
                outputs = model1(q_samples, q_captions, qqtargets, s_samples, s_captions, s_targets)
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            pred_masks = outputs["pred_masks"][0]
            pred_ref_points = outputs["reference_points"][0]
            origin_h, origin_w = int(q_targets['size'][0]), int(q_targets['size'][1])

            len_frames, origin_h1, origin_w1 = mask.shape

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


            pred_masks = torch.nn.Sigmoid()(pred_masks)

            test_evaluations.update_evl(tuple([1]), mask, pred_masks)
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
                    pred_masks = F.interpolate(pred_masks, size=(origin_h1, origin_w1), mode='bilinear', align_corners=False)
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
    return mean_j, mean_f


def show1(pre_mask):

    plt.figure("figure name screenshot")
    plt.imshow(pre_mask)
    plt.axis('on')
    plt.title('text title')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    list1 = range(1, 10)
    total_j = 0.0
    total_f = 0.0
    a = random.sample(list1, 1)
    for i in a:
        j, f = main(args, i)
        total_j += j
        total_f += f
    print('\n' + '*' * 32)
    print('group_%d_Averaged J on 5 seeds: %.4f' % (args.group, total_j / 5))
    print('group_%d_Averaged F on 5 seeds: %.4f' % (args.group, total_f / 5))
    print('*' * 32 + '\n')



