# from libs.config.DAN_config import OPTION as opt
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
# from libs.dataset.transform import TrainTransform, TestTransform
import datasets.transforms_video as T
import torchvision.transforms as T1
import util.misc as utils

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class YTVOSDataset(Dataset):
    def __init__(self, data_path=None, train=True, valid=False,
                 set_index=3, finetune_idx=None,
                 support_frame=5, query_frame=1, sample_per_class=10,
                 transforms=None, another_transform=None):
        self.train = train
        self.valid = valid
        self.set_index = set_index
        self.support_frame = support_frame
        self.query_frame = query_frame
        self.sample_per_class = sample_per_class
        self.transforms = transforms
        self.another_transform = another_transform
        self._transforms = make_coco_transforms('train', 640)
        self.test_transforms = test_transform()
        if data_path is None:
            # data_path = os.path.join('/Users/duni/Documents/Git/FS-RVOS') # 本地
            data_path = '/ssd-nvme1/duni/FS-RVOS'
        data_dir = os.path.join(data_path, 'data', 'mini-ref-youtube-vos')
        self.meta_folder = os.path.join(data_dir, 'train')
        self.img_dir = os.path.join(data_dir, 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'meta_expressions/train', 'meta_expressions.json')
        self.prepare_metas()

        print('data set index: ', set_index)  # set_index的作用是为了设置不同的fold
        self.train_list = [n + 1 for n in range(48) if n % 4 != (set_index - 1)]  # 36个做Train
        self.valid_list = [n + 1 for n in range(48) if n % 4 == (set_index - 1)]  # 12个做Test

        if train and not valid:
            self.class_list = self.train_list
        else:
            self.class_list = self.valid_list
        if finetune_idx is not None:
            self.class_list = [self.class_list[finetune_idx]]

        self.video_ids = []
        for class_id in self.class_list:
            tmp_list = self.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)  # list[list[video_id]] 获取每一个class_id对应的video_id
        if not self.train:
            self.test_video_classes = []
            for i in range(len(self.class_list)):
                for j in range(len(self.video_ids[i]) - 1):  # remove the support set
                    self.test_video_classes.append(i)
        if self.train:
            self.length = len(self.class_list) * sample_per_class
        else:
            self.length = len(self.test_video_classes)  # test 与set_index有关 345 351 362 370

    def prepare_metas(self):
        with open(os.path.join(str(self.meta_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        # read expression data
        # data/mini-ref-youtube-vos/meta_expressions/train/meta_expressions.json
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
                # get object category
                obj_id = exp_dict['obj_id']
                meta['category'] = vid_meta['objects'][obj_id]['category']
                meta['category_id'] = category_dict[vid_meta['objects'][obj_id]['category']]
                self.metas[vid].append(meta)
            """
            '1d746352a6': [
            {'video': '1d746352a6','exp_id': 0,  'exp': 'a big black and white stripped cow raging towards others in front of it', 'obj_id': 1, 
            'frames': ['00000', '00005', '00010', '00015', '00020', '00025', '00030', '00035', '00040', '00045', '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 27}, 
            {'video': '1d746352a6', 'exp_id': 1, 'exp': 'a black and white cow with horns', 'obj_id': 1, 'frames': ['00000', '00005', '00010', '00015', '00020', '00025', '00030', '00035', '00040', '00045', 
            '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 28}, 'category_id': 27}, 
            {'video': '1d746352a6', 'exp_id': 2, 'exp': 'the second cow to the left of another in the front row', 'obj_id': 2, 'frames': ['00000', '00005', '00010', '00015', '00020', '00025', '00030', 
            '00035', '00040', '00045', '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 27}, 
            {'video': '1d746352a6', 'exp_id': 3, 'exp': 'a black cow walking', 'obj_id': 2, 'frames': ['00000', '00005', '00010', '00015', '00020', '00025', '00030', '00035', '00040', '00045', 
            '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 27}, 
            {'video': '1d746352a6', 'exp_id': 4, 'exp': 'the first cow to the right of another in the front row', 'obj_id': 3, 'frames': ['00000', '00005', '00010', '00015', '00020', 
            '00025', '00030', '00035', '00040', '00045', '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 27}, 
            {'video': '1d746352a6','exp_id': 5,  'exp': 'a black cow to the far right of two other cows', 'obj_id': 3, 'frames': ['00000', '00005', '00010', '00015', '00020', '00025', '00030', 
            '00035', '00040', '00045', '00050', '00055', '00060', '00065'], 'category': 'cow', 'category_id': 27}],
            """

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
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def get_GT_byclass(self, vid, class_id, frame_num=1, test=False):
        meta = self.metas[vid]
        frame_list = meta[0]['frames']
        frame_len = len(frame_list)
        obj_exp = defaultdict(list)  # 存放obj_id其对应的exp_id
        obj_ids = []  # 存放obj_id
        for i in meta:
            obj_exp[i['obj_id']].append(i['exp_id'])
            obj_ids.append(i['obj_id'])
        obj_id = random.sample(obj_ids, 1)[0]
        exp_id = random.sample(obj_exp[obj_id], 1)[0]
        category_id = meta[0]['category_id']
        exp = meta[exp_id]['exp']
        exp = " ".join(exp.lower().split())
        choice_frame = random.sample(frame_list, 1)  # 如果为support，这个就作为最终的choice_frame
        if test:
            frame_num = frame_len  # 注意 测试时的query_set是一个视频中的所有帧
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
        imgs, labels, boxes, masks, valid, exps = [], [], [], [], [], []
        for frame_idx in choice_frame:
            img_path = os.path.join(self.meta_folder, 'JPEGImages', vid, frame_idx + '.jpg')
            mask_path = os.path.join(self.meta_folder, 'Annotations', vid, frame_idx + '.png')
            # img =np.array(Image.open(img_path).convert('RGB'))
            # mask = Image.open(mask_path).convert('P')  # 模式“P”为8位彩色图像
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('P')

            label = torch.tensor(category_id)
            mask = np.array(mask)
            mask = (mask == obj_id).astype(np.float32)  # 0,1 binary 这一步是为了只保留obj_id对应的分割目标的mask区域
            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:  # some frame didn't contain the instance
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)
            mask = torch.from_numpy(mask)
            # append
            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)
            exps.append(exp)
        # w, h = img.shape
        # w, h = mask.shape
        w, h = img.size
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        target = {
            # 'frames_idx': torch.tensor(sample_indx),  # [T,]
            'labels': labels,  # [T,]
            'boxes': boxes,  # [T, 4], xyxy
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }

        # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
        # if test:
        #     imgs = [transform(i) for i in imgs]
        #     imgs = torch.stack(imgs, dim=0).to(args.device)
        # else:
        # imgs, target = self._transforms(imgs, target)
        if self.train:
            imgs, target = self._transforms(imgs, target)
        else:
            imgs = [self.test_transforms(img) for img in imgs]
            _, h, w = imgs[0].shape
            target['size'] = torch.tensor([h, w])
        imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]

        # FIXME: handle "valid", since some box may be removed due to random crop
        if torch.any(target['valid'] == 1):  # at leatst one instance
            instance_check = True
        else:
            idx = random.randint(0, self.__len__() - 1)
        return imgs, target, choice_frame

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class  # 类别id
        vid_set = self.video_ids[list_id]
        query_vid = random.sample(vid_set, 1)
        s_vid_set = [x for x in vid_set if x not in query_vid]
        # support_vid = random.sample(vid_set, self.support_frame)  # 这里的query和support可以从同一个视频中采样，是否需要避免？
        support_vid = random.sample(s_vid_set, 1)
        q_imgs, q_targets, _ = self.get_GT_byclass(query_vid[0], self.class_list[list_id], self.query_frame)
        s_imgs, s_targets, _ = self.get_GT_byclass(support_vid[0], self.class_list[list_id], self.support_frame)
        return q_imgs, q_targets, s_imgs, s_targets, self.class_list[list_id]

    def __gettestitem__(self, idx):
        # random.seed()
        begin_new = False
        # begin_new其实只为了提取10个类别的起始id
        if idx == 0:
            begin_new = True
        else:
            if self.test_video_classes[idx] != self.test_video_classes[idx - 1]:
                begin_new = True
        list_id = self.test_video_classes[idx]  # 类别id
        vid_set = self.video_ids[list_id]

        s_imgs, s_targets = [], []
        # support_vid = random.sample(vid_set, 1)
        # support_frames, support_masks, support_exps, _, _ = self.get_GT_byclass(support_vid[0], self.class_list[list_id],
        #                                                                   self.support_frame)
        if begin_new:
            support_vid = random.sample(vid_set, 1)
            query_vids = []
            for id in vid_set:
                if not id in support_vid:
                    query_vids.append(id)
            self.query_ids = query_vids
            self.query_idx = -1
            s_imgs, s_targets, _ = self.get_GT_byclass(support_vid[0], self.class_list[list_id],
                                                       self.support_frame)
            # for i in range(self.support_frame):
            #     one_frame, one_mask, one_exp = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
            #     support_frames += one_frame
            #     support_masks += one_mask
            #     support_exps += one_exp
        self.query_idx += 1
        query_vid = self.query_ids[self.query_idx]

        q_imgs, q_targets, frames_name = self.get_GT_byclass(query_vid, self.class_list[list_id], test=True)
        vid_info = self.metas[query_vid]
        vid_name = vid_info[0]['video']
        if not begin_new:
            s_imgs, s_targets = q_imgs, q_targets
        return q_imgs, q_targets, s_imgs, s_targets, self.class_list[list_id], vid_name, begin_new, frames_name

    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __len__(self):
        return self.length

    def get_class_list(self):
        return self.class_list


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

    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def test_transform():
    return T1.Compose([
        T1.Resize(360),
        T1.ToTensor(),
        T1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    traindataset = YTVOSDataset(train=True, query_frame=5, support_frame=5)
    data_loader_train = DataLoader(traindataset, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)
    test_dataset = YTVOSDataset(train=False, query_frame=5, support_frame=5)
    data_loader_test = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=utils.collate_fn)
    trained_iter = 0
    for iter, data in enumerate(data_loader_train):
        trained_iter += 1
        qsamples, qtargets, new_samples, new_targets, idx= data
        print(qtargets)
