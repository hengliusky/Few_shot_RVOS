
The official implementation of the paper: 

<div align="center">
<h1>
<b>
Few-Shot Referring <br> Video Object Segmentation
</b>
</h1>
</div>


<p align="center"><img src="docs/network.png" width="800"/></p>

<!--
[**Language as Queries for Referring Video Object Segmentation**](https://arxiv.org/abs/2201.00487)
>
> Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, Ping Luo
-->


## Abstract
Referring video object segmentation (RVOS), as a supervised learning task, relies on sufficient annotated data for a given scene. 
However, in more realistic scenarios, only very limited annotations are available for a new scene, which poses great challenges to existing RVOS methods.
With this in mind, we introduce a novel video segmentation task - Few-Shot Referring Video Object Segmentation (FS-RVOS). 
With a few annotated video samples, FS-RVOS can quickly learn new semantic information, which allows it to adapt to different scenes. 
To foster research in this direction, we build up a new FS-RVOS benchmark based on currently available relevant datasets. 
The benchmark covers a wide range and includes multiple situations, which can maximally simulate real-world scenarios. 
To solve this task, we propose a simple yet effective model with a newly designed Cross-modal Affinity (CMA) module based on a transformer architecture. 
The CMA module can build multimodal information affinity between the support set and the query set, which can effectively learn new semantic information from a variety of annotated data.
Extensive experiments show that our model can adapt well to different scenarios by only relying on a few annotated data. 
Likewise, our model outperforms the baselines significantly on benchmarks, reaching state-of-the-art performance. 
On Mini-Ref-Youtube-VOS, our model achieves an average performance of  53.1 J and 54.8 F, which is 10% better than the baseline performance.
Furthermore, we show impressive results of 77.7 J and 74.8 F on Mini-Ref-SAIL-VOS, which are significantly better than the baseline.



<!--
## Requirements

We test the codes in the following environments, other versions may also be compatible:

- CUDA 11.1
- Python 3.7
- Pytorch 1.8.1


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data.md](docs/data.md) for data preparation.

We provide the pretrained model for different visual backbones. You may download them [here](https://drive.google.com/drive/u/0/folders/11_qps3q75aH41IYHlXToyeIBUKkfdqso) and put them in the directory `pretrained_weights`.

 
For the Swin Transformer and Video Swin Transformer backbones, the weights are intialized using the pretrained model provided in the repo [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). For your convenience, we upload the pretrained model in the google drives [swin_pretrained](https://drive.google.com/drive/u/0/folders/1QWLayukDJYAxTFk7NPwerfso3Lrx35NL) and [video_swin_pretrained](https://drive.google.com/drive/u/0/folders/19qb9VbKSjuwgxsiPI3uv06XzQkB5brYM). -->

<!--
After the organization, we expect the directory struture to be the following:

```
ReferFormer/
├── data/
│   ├── ref-youtube-vos/
│   ├── ref-davis/
│   ├── a2d_sentences/
│   ├── jhmdb_sentences/
├── davis2017/
├── datasets/
├── models/
├── scipts/
├── tools/
├── util/
├── pretrained_weights/
├── eval_davis.py
├── main.py
├── engine.py
├── inference_ytvos.py
├── inference_davis.py
├── opts.py
...
```

## Model Zoo

All the models are trained using 8 NVIDIA Tesla V100 GPU. You may change the `--backbone` parameter to use different backbones (see [here](https://github.com/wjn922/ReferFormer/blob/232b4066fb7d10845e4083e6a5a2cc0af5d1757e/opts.py#L31)).

**Note:** If you encounter the `OOM` error, please add the command `--use_checkpoint` (we add this command for Swin-L, Video-Swin-S and Video-Swin-B models).


### Ref-Youtube-VOS

To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).

| Backbone| J&F | CFBI J&F  | Pretrain | Model | Submission | CFBI Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 55.6 | 59.4 | [weight](https://drive.google.com/file/d/1mJd5zBUv4EYLOKQ0H87-NeAuInyrn577/view?usp=sharing) | [model](https://drive.google.com/file/d/1VKYIbd3tiuLyWkh7ajnIiA3HZ3_IdvxV/view?usp=sharing) | [link](https://drive.google.com/file/d/1IXKu8a06ppPAVBvy4Y0UfcKhCat4HRJt/view?usp=sharing) | [link](https://drive.google.com/file/d/1VJAKZ_j7kQFpocv_vDzER47CXWwAAE8h/view?usp=sharing) |
| ResNet-101 | 57.3 | 60.3 | [weight](https://drive.google.com/file/d/1EMOwwAygdSfTZiVxI4f0UaVd7P6JzmuM/view?usp=sharing) | [model](https://drive.google.com/file/d/1FCHAAMf-HXPhZGTZp748l3pn6FfMyV1L/view?usp=sharing) | [link](https://drive.google.com/file/d/1cFxjVW2RlwjoVYR1M6NlkRpv9L3tPlcZ/view?usp=sharing) | [link](https://drive.google.com/file/d/1RPnFPqf7iiVypc7QbN-ev6s6xfmD-m5c/view?usp=sharing) |
| Swin-T | 58.7 | 61.2 | [weight](https://drive.google.com/file/d/155sZm6yE7YQ8Y8Ln0ShaVZKLejYORqTQ/view?usp=sharing) | [model](https://drive.google.com/file/d/19jIbjRRUGDhfnI604Pw7hcGP5DqdvVtl/view?usp=sharing) | [link](https://drive.google.com/file/d/1eZZ-2zz0gdCwPrislGP3WKAHk-RnNY7v/view?usp=sharing) | [link](https://drive.google.com/file/d/1O9B35oieBfo7sRjxTpSyFz52J2AAHLce/view?usp=sharing) |
| Swin-L | 62.4 | 63.3 | [weight](https://drive.google.com/file/d/1eJKNHvk_KcFuT4k6Te7HDuuSXH2DVOY5/view?usp=sharing) | [model](https://drive.google.com/file/d/1_uwwlWv8AXhHfE8GVId7YtGraznRebaZ/view?usp=sharing) | [link](https://drive.google.com/file/d/1uxBwbKdlilaCNt-RbdcPj1LshA-WY9Q6/view?usp=sharing) | [link](https://drive.google.com/file/d/16kVmJzv5oXzk3zGcfMcb2sEiN6HTOCmW/view?usp=sharing) |
| Video-Swin-T* | 55.8 | - | - | [model](https://drive.google.com/file/d/1vNiQGpKuYfR7F7YKZK7H2HAzljDf9Wuf/view?usp=sharing) | [link](https://drive.google.com/file/d/18G0qIeZndacj3Y0EuyJsZFeFRWJ0_3O_/view?usp=sharing) | - |
| Video-Swin-T | 59.4 | - | [weight](https://drive.google.com/file/d/1g9Dm1vLdwpwSKVtIZzWKPUk2-zK3IbQa/view?usp=sharing) | [model](https://drive.google.com/file/d/17RL6o_A57giHT-bMuP7ysUGogueT7wYm/view?usp=sharing) | [link](https://drive.google.com/file/d/1nhjvDWgMWufMGAjOKesgyLRB_-Ct6kXP/view?usp=sharing) | - |
| Video-Swin-S | 60.1 | - | [weight](https://drive.google.com/file/d/1GrhFhsUidsVs7-dhY8NkVgWfBZdeit9C/view?usp=sharing) | [model](https://drive.google.com/file/d/1GrhFhsUidsVs7-dhY8NkVgWfBZdeit9C/view?usp=sharing) | [link](https://drive.google.com/file/d/1mhb0UAaJkTFYmGrwXHHJuaXVp-0BSkgm/view?usp=sharing) | - |
| Video-Swin-B | 62.9 | - |[weight](https://drive.google.com/file/d/1MJ1362zjqu-uZdXsSQH6pI1QOFqwv5lY/view?usp=sharing)  | [model](https://drive.google.com/file/d/1nw7D3C_RrKTMzwtzjo39snbYLbv73anH/view?usp=sharing) | [link](https://drive.google.com/file/d/1dAQdr2RqCxYUmOVQ4jFE-vv5zavNhz7B/view?usp=sharing) | - |

\* indicates the model is trained from scratch.

### Ref-DAVIS17

As described in the paper, we report the results using the model trained on Ref-Youtube-VOS without finetune.

| Backbone| J&F | J | F | Model | 
| :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.5 | 55.8 | 61.3 | [model](https://drive.google.com/file/d/1VKYIbd3tiuLyWkh7ajnIiA3HZ3_IdvxV/view?usp=sharing) |
| Swin-L | 60.5 | 57.6 | 63.4 | [model](https://drive.google.com/file/d/1_uwwlWv8AXhHfE8GVId7YtGraznRebaZ/view?usp=sharing) |
| Video-Swin-B | 61.1 | 58.1 | 64.1 | [model](https://drive.google.com/file/d/1nw7D3C_RrKTMzwtzjo39snbYLbv73anH/view?usp=sharing) |


### A2D-Sentences

The pretrained models are the same as those provided for Ref-Youtube-VOS.

| Backbone| Overall IoU | Mean IoU | mAP  | Pretrain | Model |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Video-Swin-T | 77.6 | 69.6 | 52.8 | [weight](https://drive.google.com/file/d/1g9Dm1vLdwpwSKVtIZzWKPUk2-zK3IbQa/view?usp=sharing) | [model](https://drive.google.com/file/d/1z-HO71IcFOZ9A6KD71wAXkbiQgKDpSp7/view?usp=sharing) \| [log](https://drive.google.com/file/d/1xjevouL3a1gHZN5KHtA07Cpa07R4T1Qi/view?usp=sharing) |
| Video-Swin-S | 77.7 | 69.8 | 53.9 | [weight](https://drive.google.com/file/d/1GrhFhsUidsVs7-dhY8NkVgWfBZdeit9C/view?usp=sharing) | [model](https://drive.google.com/file/d/1ng2FAX9J4FyQ7Bq1eeQC9Vvv1W8JZmek/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Uu72THexbtEje4aKXR7Q2Yd4zyPmQsi3/view?usp=sharing) |
| Video-Swin-B | 78.6 | 70.3 | 55.0 | [weight](https://drive.google.com/file/d/1MJ1362zjqu-uZdXsSQH6pI1QOFqwv5lY/view?usp=sharing) | [model](https://drive.google.com/file/d/1WlNjKS_Li-1KoUzuPM4MRM4b-oK2Ka7c/view?usp=sharing) \| [log](https://drive.google.com/file/d/1tH-f9_U0gY-iNfXm6GRyttJp3uvm5NQw/view?usp=sharing) |


### JHMDB-Sentences

As described in the paper, we report the results using the model trained on A2D-Sentences without finetune.

| Backbone| Overall IoU | Mean IoU | mAP  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| Video-Swin-T | 71.9 | 71.0 | 42.2 | [model](https://drive.google.com/file/d/1z-HO71IcFOZ9A6KD71wAXkbiQgKDpSp7/view?usp=sharing) |
| Video-Swin-S | 72.8 | 71.5 | 42.4 | [model](https://drive.google.com/file/d/1ng2FAX9J4FyQ7Bq1eeQC9Vvv1W8JZmek/view?usp=sharing) |
| Video-Swin-B | 73.0 | 71.8 | 43.7 | [model](https://drive.google.com/file/d/1WlNjKS_Li-1KoUzuPM4MRM4b-oK2Ka7c/view?usp=sharing) | 


## Get Started

Please see [Ref-Youtube-VOS](docs/Ref-Youtube-VOS.md), [Ref-DAVIS17](docs/Ref-DAVIS17.md), [A2D-Sentences](docs/A2D-Sentences.md) and [JHMDB-Sentences](docs/JHMDB-Sentences.md) for details.
-->

## Usage

### Preparation
Create a new directory data to store all the datasets.

1. Downlaod the Mini-Ref-YouTube-VOS dataset from [Baidu Yun] and Mini-Ref-SAIL-VOS dataset from [Baidu Yun].
2. Put the dataset in the `./data` folder.
```
data
├─ Mini-Ref-YouTube-VOS
│   ├─ meta_expressions
│   └─ train
│       ├─ Annotations
│       ├─ JPEGImages
│       └─ train.json
├─ Mini-Ref-SAIL-VOS
│   ├─ meta_expressions
│   └─ train
│       ├─ Annotations
│       ├─ JPEGImages
│       └─ train.json

```

### Training

```
./scripts/train_ytvos.sh [/path/to/output_dir] [/path/to/pretrained_weight] --backbone [backbone]  --group 1
```

### Inference

```
python test.py --dataset_file mini-ytvos --group 1
```



## Acknowledgement

This repo is based on [ReferFormer](https://github.com/wjn922/ReferFormer) and [DANet](https://github.com/scutpaul/DANet). Thanks for their wonderful works.


## Citation

```

```
