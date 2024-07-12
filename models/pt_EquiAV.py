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
import numpy as np
import torchvision.transforms as T
import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

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
class CrossAttention(nn.Module):
    def __init__(self, 
                q_dim: int,
                kv_dim: int,
                dim_head: int = 64, 
                num_heads: int = 8,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                attn_drop: float = 0.,
                norm_layer: nn.Module = nn.LayerNorm,        
    ):
        super().__init__()
        self.inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(self.inner_dim, kv_dim, bias=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, N_kv, _ = x_kv.shape
        q = self.to_q(x_q).reshape(B, len(x_q[0]), self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, N_kv, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, N_kv, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, len(x_q[0]), self.inner_dim)
        x = self.to_out(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim, aug_emb_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_aug = norm_layer(aug_emb_dim)
        self.attn = CrossAttention(aug_emb_dim, emb_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(emb_dim)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = Mlp(in_features=emb_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, emb, aug_vec):
        emb = emb.mean(dim=1).unsqueeze(1).repeat(1,len(aug_vec[0]),1) + self.drop_path(self.attn(self.norm_aug(aug_vec), emb))
        # emb = emb.mean(dim=1) + self.drop_path(self.attn(self.norm_aug(aug_vec), emb)).squeeze()
        emb = emb + self.drop_path(self.mlp(self.norm(emb)))
        emb = emb.mean(1)
        
        return emb
    
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
    """ EquiAV Model
    """

    def __init__(self, aug_size_a=24, aug_size_v=18, freqm=48, timem=192, num_mel_bins=128, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=12, num_heads=12, proj_hidden_dim=2048, inter_proj_out_dim=512, intra_proj_out_dim=512, aug_emb_dim=256,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, tr_pos=False, gpu=0, cen_mean=False, cen_num=16, **kwargs):
        super().__init__()

        self.aug_size_a = aug_size_a
        self.aug_size_v = aug_size_v
        
        self.cen_mean = cen_mean
        self.cen_num = cen_num

        self.timem = timem
        self.freqm = freqm

        if gpu == 0:
            print('EquiAV Model for pre-training')
            print('Learnable Positional Embedding: ', tr_pos)

        ## Audio & Visiual Encoder
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.embed_dim = embed_dim

        self.patch_embed_a = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * num_mel_bins / 256)

        if gpu == 0:
            print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v = norm_layer(embed_dim), norm_layer(embed_dim)
        self.norm_a_aug, self.norm_v_aug = norm_layer(embed_dim), norm_layer(embed_dim)

        self.aug_encoder_a = nn.Sequential(nn.Linear(aug_size_a, aug_emb_dim), 
                                    QuickGELU(),
                                    nn.Linear(aug_emb_dim, aug_emb_dim), 
                                    QuickGELU(),
                                    nn.Linear(aug_emb_dim, aug_emb_dim),
                                    )
        self.aug_encoder_v = nn.Sequential(nn.Linear(aug_size_v, aug_emb_dim), 
                                    QuickGELU(),
                                    nn.Linear(aug_emb_dim, aug_emb_dim), 
                                    QuickGELU(),
                                    nn.Linear(aug_emb_dim, aug_emb_dim),
                                    )

        self.cross_attn_a = CrossAttentionBlock(embed_dim, aug_emb_dim, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        self.cross_attn_v = CrossAttentionBlock(embed_dim, aug_emb_dim, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)

        self.pred_a_intra = MLPProjectionHead(in_dim=embed_dim, hidden_dim=proj_hidden_dim, out_dim=intra_proj_out_dim)
        self.pred_v_intra = MLPProjectionHead(in_dim=embed_dim, hidden_dim=proj_hidden_dim, out_dim=intra_proj_out_dim)
        
        self.pred_a_inter = MLPProjectionHead(embed_dim, proj_hidden_dim, inter_proj_out_dim)
        self.pred_v_inter = MLPProjectionHead(embed_dim, proj_hidden_dim, inter_proj_out_dim)
            
        self.initialize_weights()

        if gpu == 0:
            print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
            print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
        
        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int((self.patch_embed_v.num_patches) ** .5), cls_token=False)
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

    def sample_t_a(self,a):
        # initialize transform parameters
        # aug_size_a -> 7(SA+TS), 12(SA+TS+RRC), 21(SA+TS+RRC+CJ), 23(SA+TS+RRC+CJ+GB), 24(SA+TS+RRC+CJ+GB+HF)
        t_a_list = torch.tensor([],device=a.device)
        aug_size = 24
        for idx in range(self.cen_num):
            transform_nums = torch.zeros(aug_size,device=a.device) # default transform, colorjitter, grayscale, blur, flip

            min_scale = 0.08


            tm_value = torch.rand(1) * 192
            tm_min_value = torch.rand(1) * (1024 - tm_value)
            tm_start, tm_end = (tm_min_value.long()).squeeze(), (tm_min_value.long() + tm_value.long()).squeeze()

            fm_value = torch.rand(1) * 48
            fm_min_value = torch.rand(1) * (128 - fm_value)
            fm_start, fm_end = (fm_min_value.long()).squeeze(), (fm_min_value.long() + fm_value.long()).squeeze()


            # SpecAug
            # time masking
            if aug_size >= 7:
                if random.uniform(0, 1) < 0.8:
                    transform_nums[0] = 1 # SpecAug on/off
                    transform_nums[1] = tm_start
                    transform_nums[2] = tm_end
                    transform_nums[3] = fm_start
                    transform_nums[4] = fm_end
                    # Time Shifting
                    transform_nums[5] = 1 # Time Shifting on/off
                    ts_nu = np.random.randint(-1024, 1024)
                    transform_nums[6] = ts_nu / (1024)

            # Vision Augmentation
            # Random Resized Crop
            if aug_size >= 12:
                transform_nums[7] = 1 # RRC on/off
                img = torch.rand(3,1024,128)
                i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(min_scale, 1.0), ratio=(1.0 / 10.0, 1.5 / 8.0))
                _, num_t, num_f = img.shape 
                transform_nums[8] = i / num_f # top left
                transform_nums[9] = j / num_t
                transform_nums[10] = h / num_f
                transform_nums[11] = w / num_t   
                
            # Color jitter
            if aug_size >= 21:
                if random.uniform(0, 1) < 0.8:
                    fn_idx = torch.randperm(4)
                    transform_nums[12] = 1 # CJ on/off
                    transform_nums[13:17] = fn_idx
                    _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                    for fn_id in fn_idx:
                        if fn_id == 0:
                            transform_nums[17] = b-1
                        elif fn_id == 1:
                            transform_nums[18] = c-1
                        elif fn_id == 2:
                            transform_nums[19] = s-1
                        elif fn_id == 3:
                            transform_nums[20] = h
                else:
                    transform_nums[13:17] = torch.arange(4)

            # Gaussian Blur
            if aug_size >= 23:
                if random.uniform(0, 1) < 0.5:
                    transform_nums[21] = 1 # GB on/off
                    sigma = random.uniform(.1, 2.)
                    transform_nums[22] = sigma

            # Horizontal Flip
            if aug_size >= 24:
                if random.uniform(0, 1) < 0.5:
                    transform_nums[23] = 1 # HF on/off    

        
            transform_nums = transform_nums.unsqueeze(0)
            if idx == 0:
                t_a_list = transform_nums.float()
            else:
                t_a_list = torch.cat([t_a_list,transform_nums.float()],dim=0)

        return t_a_list


    def sample_t_v(self,v):
        # initialize transform parameters
        # aug_size_v -> 18(RRC+CJ+GB+HF+GS), 21(RRC+CJ+GB+HF+GS+FR+VF)
        t_v_list = torch.tensor([],device=v.device)
        aug_size = 18
        for idx in range(self.cen_num):
            transform_nums = torch.zeros(aug_size,device=v.device) # default transform, colorjitter, grayscale, blur, flip

            min_scale = 0.08

            if aug_size >= 16:
                transform_nums[0] = 1 # RRC on/off
                img = torch.rand(3,224,224)

                i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(min_scale, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
                width, height = 224, 224
                transform_nums[1] = i / height # top left
                transform_nums[2] = j / width
                transform_nums[3] = h / height
                transform_nums[4] = w / width

                if random.uniform(0, 1) < 0.8:
                    fn_idx = torch.randperm(4)
                    transform_nums[5] = 1 # CJ on/off
                    transform_nums[6:10] = fn_idx
                    _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                    for fn_id in fn_idx:
                        if fn_id == 0:
                            transform_nums[10] = b-1
                        elif fn_id == 1:
                            transform_nums[11] = c-1
                        elif fn_id == 2:
                            transform_nums[12] = s-1
                        elif fn_id == 3:
                            transform_nums[13] = h
                else:
                    transform_nums[6:10] = torch.arange(4)

                # Gaussian Blur
                if random.uniform(0, 1) < 0.5:
                    transform_nums[14] = 1 # GB on/off
                    sigma = random.uniform(.1, 2.)
                    transform_nums[15] = sigma

            if aug_size >= 18:
                # if not simple:
                # Horizontal Flip
                if random.uniform(0, 1) < 0.5:
                    transform_nums[16] = 1 # HF on/off

                # Grayscale
                if random.uniform(0, 1) < 0.2:
                    transform_nums[17] = 1 # GS on/off

            if aug_size >= 21:
                # Four-fold Rotation
                if random.uniform(0, 1) < 0.5:
                    transform_nums[18] = 1 # FR on/off
                    angle_idx = random.choice((0, 1, 2, 3))
                    transform_nums[19] = angle_idx

                # Vertical Flip
                if random.uniform(0, 1) < 0.5:
                    transform_nums[20] = 1 # VF on/off
            
        
            transform_nums = transform_nums.unsqueeze(0)

            if idx == 0:
                t_v_list = transform_nums.float()
            else:
                t_v_list = torch.cat([t_v_list,transform_nums.float()],dim=0)

        return t_v_list

    def project_embedding(self, a, v, a_aug, v_aug, t_a, t_v, t_a_list, t_v_list):

        if self.cen_mean:
            cross_t_a = self.cross_attn_a(a, self.aug_encoder_a(t_a_list))
            cross_t_v = self.cross_attn_v(v, self.aug_encoder_v(t_v_list))

            z_a_inv = self.pred_a_inter(cross_t_a)
            z_v_inv = self.pred_v_inter(cross_t_v)
            
        else:
            z_a_inv = self.pred_a_inter(a.mean(dim=1))
            z_v_inv = self.pred_v_inter(v.mean(dim=1))

        z_a_aug = self.pred_a_intra(a_aug.mean(dim=1))
        z_v_aug = self.pred_v_intra(v_aug.mean(dim=1))

        cross_t_a = self.cross_attn_a(a, self.aug_encoder_a(t_a.unsqueeze(1)))
        cross_t_v = self.cross_attn_v(v, self.aug_encoder_v(t_v.unsqueeze(1)))

        z_a_equi_t = self.pred_a_intra(cross_t_a)
        z_v_equi_t = self.pred_v_intra(cross_t_v)

        return z_a_inv, z_v_inv, z_a_equi_t, z_v_equi_t, z_a_aug, z_v_aug

    def forward(self, a, v, a_aug, v_aug, t_a, t_v, t_a_list=None, t_v_list=None):
        # embed patches
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        if a_aug is not None:
            a_aug = a_aug.transpose(2, 3)
            a_aug = self.patch_embed_a(a_aug)
            a_aug = a_aug + self.pos_embed_a
            a_aug = a_aug + self.modality_a

        if v_aug is not None:
            v_aug = self.patch_embed_v(v_aug)
            v_aug = v_aug + self.pos_embed_v
            v_aug = v_aug + self.modality_v

        for blk in self.blocks_a:
            a = blk(a)

            if a_aug is not None:
                a_aug = blk(a_aug)
        
        a = self.norm_a(a)
        if a_aug is not None:
            a_aug = self.norm_a_aug(a_aug)

        for blk in self.blocks_v:
            v = blk(v)

            if v_aug is not None:
                v_aug = blk(v_aug)

        v = self.norm_v(v)
        
        if v_aug is not None:
            v_aug = self.norm_v_aug(v_aug)

        z_a_inv, z_v_inv, z_a_equi_t, z_v_equi_t, z_a_aug, z_v_aug = self.project_embedding(a, v, a_aug, v_aug, t_a, t_v,t_a_list,t_v_list)
        
        return z_a_inv, z_v_inv, z_a_equi_t, z_v_equi_t, z_a_aug, z_v_aug

    def forward_feat(self, a, v):
        # embed patches
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        for blk in self.blocks_a:
            a = blk(a)
        a = self.norm_a(a)

        for blk in self.blocks_v:
            v = blk(v)
        v = self.norm_v(v)

        t_a = torch.zeros(24, device=a.device)
        t_a[13:17] = torch.arange(4)
        t_a = t_a.repeat(len(a),1)
        t_a = t_a.float()

        t_v = torch.zeros(18, device=a.device)
        t_v[6:10] = torch.arange(4)
        t_v = t_v.repeat(len(v),1)
        t_v = t_v.float()

        t_a_list = torch.tensor([],device=a.device)
        t_v_list = torch.tensor([],device=a.device)

        for i in range(len(a)):
            t_a = self.sample_t_a(a).unsqueeze(0)
            t_v = self.sample_t_v(v).unsqueeze(0)

            if i == 0:
                t_a_list = t_a
                t_v_list = t_v
            
            else:
                t_a_list = torch.cat([t_a_list,t_a],dim=0)
                t_v_list = torch.cat([t_v_list,t_v],dim=0)

        cross_t_a = self.cross_attn_a(a, self.aug_encoder_a(t_a_list))
        cross_t_v = self.cross_attn_v(v, self.aug_encoder_v(t_v_list))
        
        z_a_inv = self.pred_a_inter(cross_t_a)
        z_v_inv = self.pred_v_inter(cross_t_v)
        
        return z_a_inv, z_v_inv
