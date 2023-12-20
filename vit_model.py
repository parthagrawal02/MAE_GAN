# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from utils.patch_embed import PatchEmbed

from utils.pos_embed import get_2d_sincos_pos_embed


# Main changes img_size adjusted to 12 channel ECG signal - 12*1000
# Functions - Patchify and unpatchify
# Other functions remains the same


class VisionTransformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(12, 1000), patch_size=(1, 50), in_chans=1,
                 embed_dim=128, depth=6, num_heads=8,
                 mlp_ratio=3., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_classes = 10, global_pool=False, drop_rate = 0):
        super().__init__()
        use_fc_norm = global_pool
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.global_pool = global_pool
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # masking: length -> length * mask_ratio

        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            # outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)


class Discriminator(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(12, 1000), patch_size=(1, 50), in_chans=1,
                 embed_dim=128, depth=6, num_heads=8,
                 mlp_ratio=3., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_classes = 10, global_pool=False, drop_rate = 0):
        super().__init__()
        self.encoder = vit_1dcnn()
        self.output_shape = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.linear = nn.Linear(self.output_shape[0]*self.output_shape[1]*128, )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.linear(x))



# Model architecture as described in the paper.
def vit_1dcnn(**kwargs):
    model = VisionTransformer(
        patch_size=(1, 50), embed_dim=128, depth=6, num_heads=8,
        mlp_ratio=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# def generator(**kwargs):
#     model = VisionTransformer(
#         patch_size=(1, 50), embed_dim=128, depth=6, num_heads=8,
#         mlp_ratio=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def discriminator(**kwargs):
#     model = VisionTransformer(
#         patch_size=(1, 50), embed_dim=128, depth=6, num_heads=8,
#         mlp_ratio=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model