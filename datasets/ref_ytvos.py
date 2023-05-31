from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
import json
from collections import defaultdict
from datasets.categories import mini_ytvos_category_dict as category_dict
import torch
from torch.utils.data import DataLoader

import datasets.transforms_video as T
import torchvision.transforms as T1
import util.misc as utils
import matplotlib.pyplot as plt
import argparse
import opts
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class YTVOSBase(Dataset):
    def prepare_metas(self):
        with open(os.path.join(str(self.meta_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_metas_by_video.keys())
        catToVids = defaultdict(list)
        self.metas = defaultdict(list)
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp_id'] = int(exp_id)
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = int(exp_dict['obj_id'])
                meta['frames'] = vid_frames

                obj_id = exp_dict['obj_id']
                meta['category'] = vid_meta['objects'][obj_id]['category']
                meta['category_id'] = category_dict[vid_meta['objects'][obj_id]['category']]
                self.metas[vid].append(meta)
            catToVids[category_dict[vid_meta['objects'][obj_id]['category']]].append(vid)
        self.catToVids = catToVids

    def getVidIds(self, vidIds=[], catIds=[]):
        '''
        Get vid ids that satisfy given filter conditions.
        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        '''
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == 0:
            ids = self.videos
        else:
            ids = set(vidIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToVids[catId])
                else:
                    ids &= set(self.catToVids[catId])
        return list(ids)

    def bounding_box(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax



    def get_GT_byclass(self, vid, frame_num=1, test=False, obj_id=None, exp_id=None, mode='train'):
        meta = self.metas[vid]
        frame_list = meta[0]['frames']
        frame_len = len(frame_list)
        obj_exp = defaultdict(list)
        obj_ids = []
        for i in meta:
            obj_exp[i['obj_id']].append(i['exp_id'])
            obj_ids.append(i['obj_id'])

        if obj_id is None:
            obj_id = random.sample(obj_ids, 1)[0]
        if exp_id is None:
            exp_id = random.sample(obj_exp[obj_id], 1)[0]

        category_id = meta[0]['category_id']
        exp = meta[exp_id]['exp']
        exp = " ".join(exp.lower().split())
        choice_frame = random.sample(frame_list, 1)
        if test:
            frame_num = frame_len
        if frame_num > 1:
            if frame_num <= frame_len:
                choice_idx = frame_list.index(choice_frame[0])
                if choice_idx < frame_num:
                    begin_idx = 0
                    end_idx = frame_num
                else:
                    begin_idx = choice_idx - frame_num + 1
                    end_idx = choice_idx + 1
                choice_frame = [frame_list[n] for n in range(begin_idx, end_idx)]
            else:
                choice_frame = []
                for i in range(frame_num):
                    if i < frame_len:
                        choice_frame.append(frame_list[i])
                    else:
                        choice_frame.append(frame_list[frame_len - 1])
        imgs, labels, boxes, masks, valid, exps, mask_oris = [], [], [], [], [], [], []
        for frame_idx in choice_frame:
            img_path = os.path.join(self.meta_folder, 'JPEGImages', vid, frame_idx + '.jpg')
            mask_path = os.path.join(self.meta_folder, 'Annotations', vid, frame_idx + '.png')
            img = Image.open(img_path).convert('RGB')
            mask_ori = Image.open(mask_path).convert('P')
            mask = np.array(mask_ori)
            label = torch.tensor(category_id)

            mask = (mask == obj_id).astype(np.float32)
            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)
            mask = torch.from_numpy(mask)

            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)
            exps.append(exp)
            mask_oris.append(mask_ori)

        w, h = mask.shape
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        target = {

            'labels': labels,
            'boxes': boxes,
            'masks': masks,
            'valid': torch.tensor(valid),
            'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }


        if mode == 'test':
            imgs = [self._transforms(img) for img in imgs]
            target['size'] = torch.tensor([imgs[0].shape[-2], imgs[0].shape[-1]])
        else:
            imgs, target = self._transforms(imgs, target)
        imgs = torch.stack(imgs, dim=0)


        if torch.any(target['valid'] == 1):
            instance_check = True
        else:
            idx = random.randint(0, self.__len__() - 1)
        return imgs, target, choice_frame

    def __len__(self):
        return self.length

    def get_class_list(self):
        return self.class_list


class YTVOS_train(YTVOSBase):
    def __init__(self, data_path=None, set_index=1, support_frames=5, query_frames =5,  transforms=None, sample_per_class=100, train=True):
        """
        iterations: num of iterations per epoch for each class, if len(iterations)=1 all classes share the same value
        shots: num of samples for each class during fine-tuning
        """
        self.support_frames = support_frames
        self._transforms = transforms
        self.train = train
        data_dir = os.path.join(data_path, 'data', 'mini-ref-youtube-vos')
        self.meta_folder = os.path.join(data_dir, 'train')
        self.img_dir = os.path.join(data_dir, 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'meta_expressions/train', 'meta_expressions.json')
        self.support_vids = []
        self.prepare_metas()
        self.support_frame = support_frames
        self.query_frame = query_frames
        self.sample_per_class = sample_per_class
        self.class_list = [n + 1 for n in range(48) if n % 4 != (set_index - 1)]
        self.video_ids = []
        for class_id in self.class_list:
            tmp_list = self.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)

        self.length = len(self.class_list) * self.sample_per_class

    def __getitem__(self, idx):
        list_id = idx // self.sample_per_class
        vid_set = self.video_ids[list_id]
        query_vid = random.sample(vid_set, 1)
        s_vid_set = [x for x in vid_set if x not in query_vid]
        support_vid = random.sample(s_vid_set, 1)
        q_imgs, q_targets, _ = self.get_GT_byclass(query_vid[0], self.query_frame)
        s_imgs, s_targets, _ = self.get_GT_byclass(support_vid[0], self.support_frame)
        return q_imgs, q_targets, s_imgs, s_targets, self.class_list[list_id]



class YTVOS_test(YTVOSBase):
    def __init__(self, data_path=None, set_index=1, support_frames=5, transforms=None):
        """
        support_video_ids: used to filter out support videos from video list
        """
        self._transforms = transforms

        data_dir = os.path.join(data_path, 'data', 'mini-ref-youtube-vos')
        self.meta_folder = os.path.join(data_dir, 'train')
        self.img_dir = os.path.join(data_dir, 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'meta_expressions/train', 'meta_expressions.json')
        self.support_vids = []
        self.prepare_metas()
        self.support_frame = support_frames
        self.class_list = [n + 1 for n in range(48) if n % 4 == (set_index - 1)]

        self.query_video_ids = []
        for class_id in self.class_list:
            tmp_list = self.getVidIds(catIds=class_id)
            support_vid = random.sample(tmp_list, 1)[0]
            self.support_vids.append(support_vid)
            tmp_list.sort()
            tmp_list.remove(support_vid)
            self.query_video_ids.append(tmp_list)

        self.test_video_classes = []
        self.sample_metas = []
        for i, video_ids in enumerate(self.query_video_ids):
            for j, video_id in enumerate(video_ids):
                metas = self.metas[video_id]
                for meta in metas:
                    self.sample_metas.append(meta)
                    self.test_video_classes.append(i)

        self.length = len(self.sample_metas)

    def __getitem__(self, idx):
        begin_new = False
        if idx == 0:
            begin_new = True
        else:
            if self.test_video_classes[idx] != self.test_video_classes[idx - 1]:
                begin_new = True
        list_id = self.test_video_classes[idx]
        query_vids_set = self.query_video_ids[list_id]
        s_imgs, s_targets = [], []

        if begin_new:
            support_vid = self.support_vids[list_id]
            s_imgs, s_targets, _ = self.get_GT_byclass(support_vid, self.support_frame, mode='test')
        meta = self.sample_metas[idx]
        q_imgs, q_targets, frames_name = self.get_GT_byclass(meta['video'], 1, test=True, obj_id=meta['obj_id'], exp_id=meta['exp_id'],mode='test')

        vid_name = meta['video']

        if not begin_new:
            s_imgs, s_targets = q_imgs, q_targets
        return q_imgs, q_targets, s_imgs, s_targets, self.class_list[list_id], vid_name, begin_new, frames_name, meta['obj_id'], meta['exp_id']


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_yt_vos(stage, data_path, set_index=1, support_frames=5, query_frames=5, sample_per_class=100):
    if stage == 'train':
        transforms = make_coco_transforms('train', 640)
        yt_vos = YTVOS_train(data_path, set_index, support_frames, query_frames, transforms, sample_per_class)
    else:
        transforms = T1.Compose([
            T1.Resize(360),
            T1.ToTensor(),
            T1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        yt_vos = YTVOS_test(data_path, set_index, support_frames, transforms)

    return yt_vos

