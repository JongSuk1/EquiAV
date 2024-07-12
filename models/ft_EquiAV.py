# -*- coding: utf-8 -*-

import os
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x
    
class MLPProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = norm_layer(hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = norm_layer(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.norm3 = norm_layer(out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.act(self.norm1(self.fc1(x)))
        x = self.drop(x)
        x = self.act(self.norm2(self.fc2(x)))
        x = self.drop(x)
        x = self.norm3(self.fc3(x))

        return x

# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class MainModel(nn.Module):
    """ EquiAV Finetuning Model
    """
    def __init__(self, label_dim, num_mel_bins, drop_out, drop_path, train_mode="pretrained", img_size=224, target_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=12, num_heads=12, proj_hidden_dim=2048, proj_out_dim=512,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, tr_pos=False, gpu=0, ftmode='multimodal', inter_linear=True, head_type='linear', head_dim=512, **kwargs):
        super().__init__()
        if gpu == 0:
            print('EquiAV Model for fine-tunining')
            print('Learnable Positional Embedding: ', tr_pos)

        ## Audio & Visiual Encoder
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block


        self.embed_dim = embed_dim
        self.audio_length = target_length

        self.ftmode = ftmode
        self.att_softmax = nn.Softmax(dim=-1)
        
        self.patch_embed_a = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(self.audio_length * num_mel_bins / 256)
        # self.patch_embed_v.num_patches = self.patch_embed_v.num_patches * 8

        if gpu == 0:
            print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, drop=drop_out, drop_path=drop_path, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, drop=drop_out, drop_path=drop_path, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        
        # independent normalization layer for audio, visual, and audio-visual
        # before projector
        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)

        # intra-modal(equivariant) and inter-modal(invariant) projection heads
        if inter_linear:
            self.pred_a_inter = nn.Linear(embed_dim, proj_out_dim)
            self.pred_v_inter = nn.Linear(embed_dim, proj_out_dim)
        else:
            self.pred_a_inter = MLPProjectionHead(in_dim=embed_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim)
            self.pred_v_inter = MLPProjectionHead(in_dim=embed_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim)

        if self.ftmode =='multimodal':
            if head_type == 'linear':
                self.mlp_head = nn.Sequential(nn.Linear(proj_out_dim*2, label_dim))
            elif head_type == 'mlp':
                self.mlp_head = nn.Sequential(nn.Linear(proj_out_dim*2, head_dim),
                                                nn.SyncBatchNorm(head_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(head_dim, label_dim)
                                                )
        elif self.ftmode == 'audio_only' or self.ftmode == 'video_only':
            if head_type == 'linear':
                self.mlp_head = nn.Sequential(nn.Linear(embed_dim, label_dim))
            elif head_type == 'mlp':
                self.mlp_head = nn.Sequential(nn.Linear(proj_out_dim, head_dim),
                                                nn.SyncBatchNorm(head_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(head_dim, label_dim)
                                                )
        else:
            print('not appropriate train mode')
            self.mlp_head = nn.Sequential(nn.Linear(1, label_dim))
        self.initialize_weights()

        if gpu == 0:
            print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
            print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        # pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], 8, int(self.patch_embed_v.num_patches/8), cls_token=False)
        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v):
        if self.ftmode == "multimodal":
            a = a.transpose(2, 3) # batch x 1 x mel bins x target_length
            a = self.patch_embed_a(a) # batch x 512 x emb_dim
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            # audio and visual stream, independent blocks
            for blk in self.blocks_a:
                a = blk(a)
            a = self.norm_a(a)

            for blk in self.blocks_v:
                v = blk(v)
            v = self.norm_v(v)
            
            a = a.mean(dim=1)
            v = v.mean(dim=1)

            z_a_inv = self.pred_a_inter(a)
            z_v_inv = self.pred_v_inter(v)
            
            # have to distinguish the mode (multimodal, audio-inv only, etc)
            x = torch.cat((z_a_inv, z_v_inv), dim=-1)
            
            x = self.mlp_head(x)

            return x
        
        elif self.ftmode == "audio_only":
            a = a.transpose(2, 3) # batch x 1 x mel bins x target_length
            a = self.patch_embed_a(a) # batch x 512 x emb_dim
            a = a + self.pos_embed_a
            a = a + self.modality_a

            # audio and visual stream, independent blocks
            for blk in self.blocks_a:
                a = blk(a)
            a = self.norm_a(a)

            z_a_inv = self.pred_a_inter(a)

            x = z_a_inv.mean(dim=1)
            x = self.mlp_head(x)

            return x
        
        elif self.ftmode == "video_only":
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v


            for blk in self.blocks_v:
                v = blk(v)
            v = self.norm_v(v)

            # z_v_inv = self.pred_v_inter(v)

            # x = z_v_inv.mean(dim=1)
            # x = self.mlp_head(x)
            x = self.mlp_head(v.mean(dim=1))

            return x