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

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])  


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BASECMA(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, mask_dim, dim_feedforward,
                 controller_layers, dynamic_mask_channels,
                 aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, rel_coord=True,
                 use_dab=True,
                 num_patterns=0,
                 use_self_attn=True,
                 random_refpoints_xy=False,):

        super().__init__()
        self.num_queries = num_queries  
        self.transformer = transformer
        hidden_dim = transformer.d_model  
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab  
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        self.two_stage = two_stage
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    
                    self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))  
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.mask_dim = mask_dim
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert self.two_stage == False, "args.two_stage must be false!"

        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers  
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        
        self.rel_coord = rel_coord
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]
        self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim,
                                                  mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")
        self.controller_layers = controller_layers
        self.in_channels = mask_dim
        self.dynamic_mask_channels = dynamic_mask_channels
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
                weight_nums.append(self.dynamic_mask_channels * 1)  
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)
        encoder_dim = 256
        h_encdim = int(encoder_dim/2)
        self.support_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)
        self.query_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)
        self.conv_q = nn.Conv2d(encoder_dim,h_encdim,kernel_size=1, stride=1,padding=0)
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(128)
        self.use_self_attn = use_self_attn



    def forward(self, q_samples: NestedTensor, q_captions, q_targets, s_samples: NestedTensor, s_captions, s_targets):
        if not isinstance(q_samples, NestedTensor):
            q_samples = nested_tensor_from_videos_list(q_samples)
        if not isinstance(s_samples, NestedTensor):
            s_samples = nested_tensor_from_videos_list(s_samples)
        q_features, q_pos = self.backbone(q_samples)
        s_features, s_pos = self.backbone(s_samples)
        b = len(q_captions)
        t = q_pos[0].shape[0] // b
        query_text_features, query_text_sentence_features = self.forward_text(q_captions, device=q_pos[0].device)
        qsrcs = []
        qmasks = []
        qposes = []
        query_text_pos = self.text_pos(query_text_features).permute(2, 0, 1)
        query_text_word_features, query_text_word_masks = query_text_features.decompose()
        query_text_word_features = query_text_word_features.permute(1, 0, 2)

        
        for l, (feat, pos_l) in enumerate(zip(q_features[-3:], q_pos[-3:])):
            src, mask = feat.decompose()
            qsrc_proj_l = self.input_proj[l](src)
            n, c, h, w = qsrc_proj_l.shape

            
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
            _len_srcs = len(q_features) - 1  
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](q_features[-1].tensors)
                else:
                    src = self.input_proj[l](qsrcs[-1])
                m = rearrange(q_samples.mask, 'b t h w -> (b t) h w', b=b, t=t)
                
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                
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
        
        
        
        s_b = len(s_captions)
        s_t = s_pos[0].shape[0] // b

        support_text_features, support_text_sentence_features = self.forward_text(s_captions, device=s_pos[0].device)

        
        ssrcs = []
        smasks = []
        sposes = []

        support_text_pos = self.text_pos(support_text_features).permute(2, 0, 1)  
        support_text_word_features, support_text_word_masks = support_text_features.decompose()
        support_text_word_features = support_text_word_features.permute(1, 0, 2)  

        
        for l, (feat, pos_l) in enumerate(zip(s_features[-3:], s_pos[-3:])):
            src, mask = feat.decompose()
            
            
            ssrc_proj_l = self.input_proj[l](src)
            n, c, h, w = ssrc_proj_l.shape
            
            ssrc_proj_l = rearrange(ssrc_proj_l, '(b t) c h w -> (t h w) b c', b=s_b, t=s_t)
            ssrc_proj_l = extended_simple_transform(ssrc_proj_l, 1.3)
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
            _len_srcs = len(s_features) - 1  
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](s_features[-1].tensors)
                else:
                    src = self.input_proj[l](ssrcs[-1])
                
                m = rearrange(s_samples.mask, 'b t h w -> (b t) h w', b=s_b, t=s_t)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                
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


        
        srcs = []
        for i in range(s_b):
            support_masks = s_targets[i]['masks'].unsqueeze(1)
        for i, (query_feat, support_feat) in enumerate(zip(qsrcs, ssrcs)):
            support_feat = F.interpolate(support_feat, query_feat.size()[2:], mode='bilinear', align_corners=True)
            support_mask = F.interpolate(support_masks.float(), support_feat.size()[2:], mode='bilinear', align_corners=True)
            support_fg_feat = support_feat * support_mask
            bf, c, h, w = query_feat.shape
            after_transform = query_feat + support_fg_feat[:bf]
            srcs.append(after_transform)


        if self.two_stage:  
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_embed = self.tgt_embed.weight  
                refanchor = self.refpoint_embed.weight  
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                
                tgt_embed = self.tgt_embed.weight  
                pat_embed = self.patterns_embed.weight  
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1)  
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1)  
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)  
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        else:
            query_embeds = self.query_embed.weight

        text_embed = repeat(query_text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, qmasks, qposes, query_embeds)  

        out = {}
        
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
            outputs_coord = tmp.sigmoid()  
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  
        out['pred_boxes'] = outputs_coord[-1]  


        mask_features = self.pixel_decoder(q_features, query_text_features, q_pos, memory,
                                           nf=t)
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)
        
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references,
                                                             q_targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)

        if not self.training:
            
            inter_references = inter_references[-2, :, :, :2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t)
            out['reference_points'] = inter_references  
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):

        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)  
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)
            text_sentence_features = encoded_text.pooler_output
            text_sentence_features = self.resizer(text_sentence_features)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):

        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  

        
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        
        reference_points = new_reference_points

        
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                              locations.reshape(1, 1, 1, h, w, 2)  
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3,
                                                      4)  

            
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  
        mask_features = mask_features.reshape(1, -1, h, w)

        
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  

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
        
        
        
        B, CQ, WQ = Q.shape
        _, CV, WK = V.shape

        P = torch.bmm(K, Q)  
        P = P / math.sqrt(CQ)
        P = torch.softmax(P, dim=1)

        M = torch.bmm(V, P)  

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
            
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            
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
        
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def simple_transform(x, beta):
    x = torch.sign(x) / torch.pow(torch.log(1 / abs(x) + 1), beta)
    return x

def extended_simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1), beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1), beta)
    return x_pos+x_neg

class QueryKeyValue(nn.Module):
    
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
            num_classes = 91  
    device = torch.device(args.device)

    
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = BASECMA(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,  
        num_feature_levels=args.num_feature_levels,  
        num_frames=args.num_frames,  
        mask_dim=args.mask_dim,  
        dim_feedforward=args.dim_feedforward,  
        controller_layers=args.controller_layers,  
        dynamic_mask_channels=args.dynamic_mask_channels,  
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,  
        freeze_text_encoder=args.freeze_text_encoder,  
        rel_coord=args.rel_coord,
        use_dab=args.use_dab,
        num_patterns=args.num_patterns,  
        use_self_attn=args.use_self_attn,
        random_refpoints_xy=args.random_refpoints_xy
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:  
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    
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

    
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
