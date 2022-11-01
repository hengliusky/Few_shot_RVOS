import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)  # 与dab不同
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_support_names', default=['support_qkv', 'query_qkv', 'conv_q'], type=str, nargs='+')

    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)  # 与dab不同
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')



    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.") 

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true') # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true', # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # add
    parser.add_argument('--pe_temperatureH', default=20, type=int,
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int,
                        help="Temperature for width positional encoding.")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # add
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true',
                        help="Random init the x,y of anchor boxes and freeze them.")


    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true') # default: False

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int, 
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int, 
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=5, type=float,
                        help="mask coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name') 
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)  # 可以设小一点，减少显存开销

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--ngpu', default=3, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')
    parser.add_argument('--cat_id', default=1, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # few-show ReferFormer
    parser.add_argument('--iterations_per_epoch', default=[20], help='num of iterations for each class')
    parser.add_argument('--shots', default=1)
    parser.add_argument('--data_path', default='/ssd-nvme1/duni/FS-RVOS')
    parser.add_argument('--support_frames', default=5, help='num of frames for each training clip')
    parser.add_argument('--query_frames', default=5, help='num of frames for each training clip')
    parser.add_argument("--sample_per_class", type=int, default=100)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--use_dab", default=True, type=bool)
    parser.add_argument("--use_self_attn", default=True, type=bool)
    return parser



