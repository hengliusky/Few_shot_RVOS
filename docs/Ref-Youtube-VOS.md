## Ref-Youtube-VOS

### Model Zoo

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

### Inference & Evaluation


First, inference using the trained model.

```
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir=[/path/to/output_dir] --resume=[/path/to/model_weight] --backbone [backbone] 
```

```
python3 inference_ytvos.py --visualize --with_box_refine --binary --freeze_text_encoder --output_dir=ytvos_dirs/swin_tiny --resume=p_model/ytvos_swin_tiny.pth --backbone swin_t_p4w7
```

If you want to visualize the predited masks, you may add `--visualize` to the above command.

Then, enter the `output_dir`, rename the folder `valid` as `Annotations`. Use the following command to zip the folder:

```
zip -q -r submission.zip Annotations
```

To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).

### Training

The following command includes the training and inference stages.

```
./scripts/dist_train_test_ytvos.sh [/path/to/output_dir] [/path/to/pretrained_weight] --backbone [backbone] 
```

For example, training the Video-Swin-Tiny model, run the following command:

```
./scripts/dist_train_test_ytvos.sh ytvos_dirs/video_swin_tiny pretrained_weights/video_swin_tiny_pretrained.pth --backbone video_swin_t_p4w7 
```
```
./scripts/dist_train_test_ytvos.sh sailvos/r50 pretrained_weights/ytvos_r50_joint.pth --backbone resnet50 
```