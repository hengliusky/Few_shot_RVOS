#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=True)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, inplane1, inplane2, mdim):  # 默认2048,1024,256
        super(Decoder, self).__init__()
        self.convFM1 = nn.Conv2d(inplane1, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        # self.convFM2 = nn.Conv2d(int(inplane/2), mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF4 = Refine(inplane2, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1
        input_proj_list = []
        hidden_dim = 256
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(512, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        ))
        self.input_pro = nn.ModuleList(input_proj_list)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(256, 256, 512, 3)
        self.fusion = VisionLanguageFusionModule(d_model=256, nhead=8)
        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, sq_srcs, feature, texts, f, use_text=False):
        features = []
        for i, feat in enumerate(feature):
            x, x_mask = feat.decompose()
            features.append(x)
        bf = f.shape[0]
        b = texts[0].shape[1]
        t = int(bf/b)
        m5 = self.ResMM(self.convFM1(sq_srcs[1]))  # b*f,256,h/32,w/32
        # m4 = self.ResMM(self.convFM2(features[0])) # b*f,256,h/16,w/16
        m4 = self.RF4(sq_srcs[0], m5)  # b*f,256,h/16,w/16
        # f3 = self.fusion[1](query_text, features[1])  # b*f,512,h/8,w/8


        if not use_text:
            m3 = self.RF3(features[1].tensors[:bf], m4)  # out: 1/8, 256 # b*f,256,h/8,w/8
        else:
            _, _, h1, w1 = features[1].shape
            q = self.input_pro[0](features[1][:bf])  # c:512->256
            q = rearrange(q, '(b t) c h w -> (t h w) b c', b=b, t=t)
            f3 = self.fusion(tgt=q,  # c =256
                             memory=texts[0],
                             memory_key_padding_mask=texts[1],
                             pos=texts[2],
                             query_pos=None)
            f3 = f3 + self.dropout(f3)
            f3 = self.norm(f3)
            f3 = self.ffn(f3)  # b*f,512,h/8,w/8
            f3 = rearrange(f3, '(t h w) b c -> (b t) c h w', t=t, h=h1, w=w1)
            m3 = self.RF3(f3, m4)  # out: 1/8, 256 # b*f,256,h/8,w/8
        # f2 = self.fusion(query_text, features[0])
        if not use_text:
            m2 = self.RF2(features[0][:bf], m3)  # out: 1/4, 256 # b*f,256,h/4,w/4
        else:
            _, _, h0, w0 = features[0].shape
            q = rearrange(features[0][:bf], '(b t) c h w -> (t h w) b c', b=b, t=t)
            f2 = self.fusion(tgt=q,  # b*f,256,h/4,w/4
                             memory=texts[0],
                             memory_key_padding_mask=texts[1],
                             pos=texts[2],
                             query_pos=None)
            f2 = f2 + self.dropout(f2)
            f2 = self.norm(f2)
            f2 = rearrange(f2, '(t h w) b c -> (b t) c h w', t=t, h=h0, w=w0)
            m2 = self.RF2(f2, m3)  # out: 1/4, 256 # b*f,256,h/4,w/4

        # p2 = self.pred2(F.relu(m2))  # b*f,1,h/4,w/4
        p2 = F.relu(m2)
        # p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=True)  # b*f,1,h,w
        return p2


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)

        return x



if __name__ == '__main__':
    # decoder = Decoder(inplane=2048, mdim=256)
    # f0 = torch.FloatTensor(10, 1024, 16, 27)
    # f1 = torch.FloatTensor(10, 2048, 8, 14)
    # features = [f0, f1]
    # r3 = torch.FloatTensor(10, 512, 31, 54)
    # r2 = torch.FloatTensor(10, 256, 61, 107)
    # query_text = torch.FloatTensor(17, 2, 256)
    # f = torch.FloatTensor(10, 3, 241, 425)
    # pred = decoder(features, r3, r2, query_text, f)
    img = torch.FloatTensor(10, 3, 241, 425)
    mask = torch.FloatTensor(10, 1, 241, 425)