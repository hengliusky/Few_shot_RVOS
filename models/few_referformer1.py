"""
ReferFormer model class.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .test_decoder import Decoder
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])  # 拷贝函数主要是实现模型的深拷贝，将其拷贝N次。


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class ReferFormer(nn.Module):
    """ This is the ReferFormer module that performs referring video object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, mask_dim, dim_feedforward,
                 controller_layers, dynamic_mask_channels,
                 aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, rel_coord=True,
                 use_dab=True,
                 num_patterns=0,
                 random_refpoints_xy=False, ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         ReferFormer can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)


            num_classes=num_classes, ytovs:65
            num_queries=args.num_queries,  # 5
            num_feature_levels=args.num_feature_levels,   # 4
            num_frames=args.num_frames,  # 5
            mask_dim=args.mask_dim,  # 256
            dim_feedforward=args.dim_feedforward,  # 2048
            controller_layers=args.controller_layers,  # 3
            dynamic_mask_channels=args.dynamic_mask_channels,  # 8
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,  # false
            freeze_text_encoder=args.freeze_text_encoder,  # default: false
            rel_coord=args.rel_coord,
            use_dab=True,
            num_patterns=args.num_patterns,  # 0
            random_refpoints_xy=args.random_refpoints_xy

        """
        super().__init__()
        self.num_queries = num_queries  # 5
        self.transformer = transformer
        hidden_dim = transformer.d_model  # 256
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)  # 最终的类别预测层
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 最终预测box的曾
        self.num_feature_levels = num_feature_levels  # 4 # 使用的backbone特征层数，如果大于backbone提供的stage数，则使用卷积继续推进
        self.use_dab = use_dab  # true
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.two_stage = two_stage
        # Build Transformer
        # NOTE: different deformable detr, the query_embed out channels is
        # hidden_dim instead of hidden_dim * 2
        # This is because, the input to the decoder is text embedding feature
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 产生num_queries个长度为hid_dim的可学习编码向量
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)  # Embedding(5, 256)
                self.refpoint_embed = nn.Embedding(num_queries, 4)  # Embedding(5, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        # follow deformable-detr, we use the last three stages of backbone
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])  # 3
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]  # 512 1024 2048
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))  # 将不同stage的输出通道映射到相同大小
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x  4-3
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim  # 使用一层卷积层构建后续的特征金字塔
            self.input_proj = nn.ModuleList(input_proj_list)
        else:  # 否则仅使用最后一层的特征
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames  # 5
        self.mask_dim = mask_dim  # 256
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.aux_loss = aux_loss  # 使用辅助loss，即每一层decoder均使用loss约束
        self.with_box_refine = with_box_refine  # 是否使用box refine，即上一层的box作为当前层的位置编码
        assert self.two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)  # 使用val的值填充tensor
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)  # 初始化
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers  # 4
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Build Text Encoder
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        # Build FPN Decoder
        self.rel_coord = rel_coord
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]  # [256, 256, 256, 256]
        # self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim,
        #                                           mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")

        # Build Dynamic Conv
        self.controller_layers = controller_layers  # 3
        self.in_channels = mask_dim  # 256
        self.dynamic_mask_channels = dynamic_mask_channels  # 8
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)  # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums  # rel_coord为True：[2064, 64, 8] 否则为 [2048, 64, 8]
        self.bias_nums = bias_nums  # [8, 8, 1]
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)  # True为2153 否则为 2137

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)  # 3个线性层
        """
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): Linear(in_features=256, out_features=256, bias=True)
        (2): Linear(in_features=256, out_features=2137, bias=True)
        """
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)

        encoder_dim = 256
        h_encdim = int(encoder_dim/2)
        self.support_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)
        self.query_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)

        self.conv_q = nn.Conv2d(encoder_dim,h_encdim,kernel_size=1, stride=1,padding=0)
        ffn_list = []
        for i in range(2):
            ffn_list.append(MLP(hidden_dim, hidden_dim, 1024 * (i + 1), 3))
        self.ffn = nn.ModuleList(ffn_list)
        self.Decoder = Decoder(2048, 1024, 256)



    def forward(self, q_samples: NestedTensor, q_captions, q_targets, s_samples: NestedTensor, s_captions, s_targets):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # Backbone
        # 将样本转为NestedTensor。isinstance() 函数来判断一个对象是否是一个已知的类型
        if not isinstance(q_samples, NestedTensor):
            q_samples = nested_tensor_from_videos_list(q_samples)
        if not isinstance(s_samples, NestedTensor):
            s_samples = nested_tensor_from_videos_list(s_samples)
            # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        # 输入到CNN backbone提取特征   pos为位置编码
        q_features, q_pos = self.backbone(q_samples)  # q_pos:bf,256,h,w;  q_features: bf,256/512/1024/2048, h,w
        s_features, s_pos = self.backbone(s_samples)

        b = len(q_captions)
        t = q_pos[0].shape[0] // b


        query_text_features, query_text_sentence_features = self.forward_text(q_captions, device=q_pos[0].device)

        # prepare vision and text features for transformer
        qsrcs = []
        qmasks = []
        qposes = []

        query_text_pos = self.text_pos(query_text_features).permute(2, 0, 1)  # [length, batch_size, c]
        query_text_word_features, query_text_word_masks = query_text_features.decompose()
        query_text_word_features = query_text_word_features.permute(1, 0, 2)  # [length, batch_size, c]

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(q_features[-3:], q_pos[-3:])):
            src, mask = feat.decompose()
            qsrc_proj_l = self.input_proj[l](src)
            n, c, h, w = qsrc_proj_l.shape

            # vision language early-fusion
            qsrc_proj_l = rearrange(qsrc_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            qsrc_proj_l = self.fusion_module(tgt=qsrc_proj_l,
                                            memory=query_text_word_features,
                                            memory_key_padding_mask=query_text_word_masks,
                                            pos=query_text_pos,
                                            query_pos=None
                                            )
            qsrc_proj_l = rearrange(qsrc_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            qsrcs.append(qsrc_proj_l)
            qmasks.append(mask)
            qposes.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(q_features) - 1):
            _len_srcs = len(q_features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](q_features[-1].tensors)
                else:
                    src = self.input_proj[l](qsrcs[-1])
                m = rearrange(q_samples.mask, 'b t h w -> (b t) h w', b=b, t=t)
                # m = q_samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                src = self.fusion_module(tgt=src,
                                         memory=query_text_word_features,
                                         memory_key_padding_mask=query_text_word_masks,
                                         pos=query_text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                qsrcs.append(src)
                qmasks.append(mask)
                qposes.append(pos_l)
        # qsrcs: bf, 256, h/8, h/16, h/32, h/64
        # qmasks: bf, h/8, h/16, h/32, h/64
        # qposes: bf, 256, h / 8, h / 16, h / 32, h / 64
        s_b = len(s_captions)
        s_t = s_pos[0].shape[0] // b

        support_text_features, support_text_sentence_features = self.forward_text(s_captions, device=s_pos[0].device)

        # prepare vision and text features for transformer
        ssrcs = []
        smasks = []
        sposes = []

        support_text_pos = self.text_pos(support_text_features).permute(2, 0, 1)  # [length, batch_size, c]
        support_text_word_features, support_text_word_masks = support_text_features.decompose()
        support_text_word_features = support_text_word_features.permute(1, 0, 2)  # [length, batch_size, c]

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(s_features[-3:], s_pos[-3:])):
            src, mask = feat.decompose()
            ssrc_proj_l = self.input_proj[l](src)
            n, c, h, w = ssrc_proj_l.shape

            # vision language early-fusion
            ssrc_proj_l = rearrange(ssrc_proj_l, '(b t) c h w -> (t h w) b c', b=s_b, t=s_t)
            ssrc_proj_l = self.fusion_module(tgt=ssrc_proj_l,
                                             memory=support_text_word_features,
                                             memory_key_padding_mask=support_text_word_masks,
                                             pos=support_text_pos,
                                             query_pos=None
                                             )
            ssrc_proj_l = rearrange(ssrc_proj_l, '(t h w) b c -> (b t) c h w', t=s_t, h=h, w=w)

            ssrcs.append(ssrc_proj_l)
            smasks.append(mask)
            sposes.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(s_features) - 1):
            _len_srcs = len(s_features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](s_features[-1].tensors)
                else:
                    src = self.input_proj[l](ssrcs[-1])
                # m = s_samples.mask
                m = rearrange(s_samples.mask, 'b t h w -> (b t) h w', b=s_b, t=s_t)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=s_b, t=s_t)
                src = self.fusion_module(tgt=src,
                                         memory=support_text_word_features,
                                         memory_key_padding_mask=support_text_word_masks,
                                         pos=support_text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=s_t, h=h, w=w)

                ssrcs.append(src)
                smasks.append(mask)
                sposes.append(pos_l)
        for i in range(s_b):
            support_mask = s_targets[i]['masks'].unsqueeze(1)  # bf, 1,h,w
        srcs = []
        for i, (query_feat, support_feat ) in enumerate(zip(qsrcs, ssrcs)):
            support_feat = F.interpolate(support_feat, query_feat.size()[2:], mode='bilinear', align_corners=True)
            support_mask = F.interpolate(support_mask.float(), support_feat.size()[2:], mode='bilinear', align_corners=True).to(torch.bool)
            # query_feat = F.interpolate(query_feat, support_feat.size()[2:], mode='bilinear', align_corners=True)
            support_fg_feat = support_feat * support_mask
            # support_bg_feat = support_feat * (1 - support_mask)

            # q、k、v bf,128,h,w
            _, support_k, support_v = self.support_qkv(support_fg_feat)
            query_q, query_k, query_v = self.query_qkv(query_feat)
            _, _, qh, qw = query_k.shape
            _, c, h, w = support_k.shape
            _, vc, _, _ = support_v.shape

            assert qh == h and qw == w
            # transforms query_middle_q to support_kv
            # support [b*f c h w] -> [b f c h w] -> [b c f h w] -> [b c WF]
            support_k = support_k.view(s_b, s_t, c, h, w)
            support_v = support_v.view(s_b, s_t, vc, h, w)
            # B, WK, CK
            support_k = support_k.permute(0, 2, 1, 3, 4).contiguous().view(b, c, -1).permute(0, 2, 1).contiguous()
            # B, CV, WK
            support_v = support_v.permute(0, 2, 1, 3, 4).contiguous().view(b, vc, -1)
            middle_frame_index = int(t / 2)
            query_q = query_q.view(b, t, c, h, w)
            query_k = query_k.view(b, t, c, h, w)
            middle_q = query_q[:, middle_frame_index]
            assert len(middle_q.shape) == 4
            # B, CQ, WQ
            middle_q = middle_q.view(b, c, -1)
            # B CV WQ --> V
            new_V, sim_refer = self.transformer1(middle_q, support_k, support_v)
            # print(sim_refer.shape)
            # transform query_qkv to query_middle_kv
            # B WK CK
            middle_K = query_k[:, middle_frame_index]
            middle_K = middle_K.view(b, c, -1).permute(0, 2, 1).contiguous()
            query_q = query_q.permute(0, 2, 1, 3, 4).contiguous().view(b, c, -1)
            Out, sim_middle = self.transformer1(query_q, middle_K, new_V)
            after_transform = Out.view(b, vc, t, h, w)
            after_transform = after_transform.permute(0, 2, 1, 3, 4).contiguous()

            # [batch*frames, 1024, h/16,w/16]
            query_feat = self.conv_q(query_feat)
            after_transform = after_transform.view(-1, vc, h, w)
            after_transform = torch.cat((after_transform, query_feat), dim=1)
            srcs.append(after_transform)
        # Transformer
        # query_embeds = self.query_embed.weight  # [num_queries, c]
        ###20220901修改
        # for i, (query_feat, support_feat) in enumerate(zip(qsrcs, ssrcs)):
        #     support_feat = F.interpolate(support_feat, query_feat.size()[2:], mode='bilinear', align_corners=True)
        #     # support_mask = F.interpolate(support_mask.float(), support_feat.size()[2:], mode='bilinear', align_corners=True)
        #     n, c, h, w = query_feat.shape
        #     query_feat = rearrange(query_feat, '(b t) c h w -> (t h w) b c', b=b, t=t)
        #     support_feat = rearrange(support_feat, '(b t) c h w -> (t h w) b c', b=s_b, t=s_t)
        #     src = self.fusion_module(tgt=query_feat,
        #                          memory=support_feat,
        #                          memory_key_padding_mask=None,
        #                          pos=None,
        #                          query_pos=None
        #                          )
        #     src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
        #     srcs.append(src)
        if self.two_stage:  # false
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_embed = self.tgt_embed.weight  # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                # multi patterns
                tgt_embed = self.tgt_embed.weight  # nq, 256
                pat_embed = self.patterns_embed.weight  # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1)  # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1)  # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)  # nq*num_pat, 4
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight

        text_embed = repeat(query_text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, qmasks, qposes, query_embeds)  # 这里的src是经过视觉、文本特征融合之后的特征
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi] c=256
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]



        out = {}
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]


        # Segmentation
        
        # mask_features = self.pixel_decoder(q_features, query_text_features, q_pos, memory,
        #                                    nf=t)  # [batch_size*time, c, out_h, out_w]
        # mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)

        #### 0908
        texts = [query_text_word_features, query_text_word_masks, query_text_pos]
        mems = []
        for i, mem in enumerate(memory[1:]):
            _, c, h, w = mem.shape
            mem = rearrange(mem, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mem = self.ffn[i](mem)
            mem = rearrange(mem, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mems.append(mem)
        f = rearrange(q_samples.tensors, 'b t c h w -> (b t) c h w', b=b, t=t)
        pred = self.Decoder(mems, q_features[:2], texts, f, use_text=True)
        # print(pred.shape)
        mask_features = rearrange(pred, '(b t) c h w -> b t c h w', b=b, t=t)

        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])  # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references,
                                                             q_targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)

        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t)
            out['reference_points'] = inter_references  # the reference points of last layer input
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state  # 类型为tensor，它是模型最后一层输出的隐藏状态。
            text_features = self.resizer(text_features)  # 将通道数由768改为256
            text_masks = text_attention_mask
            # # 去除首尾的token
            # length = text_features.shape[1]
            # text_features = text_features[:, 1:length - 1]
            # text_masks = text_masks[:, 1:length - 1]
            text_features = NestedTensor(text_features, text_masks)  # NestedTensor

            text_sentence_features = encoded_text.pooler_output  # 类型为tensor，这是序列的第一个token：[CLS]的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。
            # 这个输出不是对输入的语义内容的一个很好的总结，对于整个输入序列的隐藏状态序列的平均化或池化通常更好。
            text_sentence_features = self.resizer(text_sentence_features)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                              locations.reshape(1, 1, 1, h, w, 2)  # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3,
                                                      4)  # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def transformer1(self, Q, K, V):
        # Q : B CQ WQ
        # K : B WK CQ
        # V : B CV WK
        B, CQ, WQ = Q.shape
        _, CV, WK = V.shape

        P = torch.bmm(K, Q)  # B WK WQ  矩阵乘法，第一维相等，然后第一个数组的第三维和第二个数组的第二维度要求一样
        P = P / math.sqrt(CQ)
        P = torch.softmax(P, dim=1)

        M = torch.bmm(V, P)  # B CV WQ

        return M, P

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class QueryKeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(indim, keydim,kernel_size=3, padding=1,stride=1)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.query(x),self.Key(x), self.Value(x)



def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 36
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91  # for coco 91
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = ReferFormer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,  # 5
        num_feature_levels=args.num_feature_levels,  # 4
        num_frames=args.num_frames,  # 5
        mask_dim=args.mask_dim,  # 256
        dim_feedforward=args.dim_feedforward,  # 2048
        controller_layers=args.controller_layers,  # 3
        dynamic_mask_channels=args.dynamic_mask_channels,  # 8
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,  # false
        freeze_text_encoder=args.freeze_text_encoder,  # default: false
        rel_coord=args.rel_coord,
        use_dab=True,
        num_patterns=args.num_patterns,  # 0
        random_refpoints_xy=args.random_refpoints_xy
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:  # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    # losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
