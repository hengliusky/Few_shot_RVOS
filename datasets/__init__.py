import torch.utils.data
import torchvision

from .ytvos import build as build_ytvos





def get_coco_api_from_dataset(dataset):
    for _ in range(10):


        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ytvos':
        return build_ytvos(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')


