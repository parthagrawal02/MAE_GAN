# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pyto/Users/parthagrawal02/Library/CloudStorage/GoogleDrive-acads.parth@gmail.com/My Drive/project_folder/ECG_MAE_code/main_pretrain.pyrch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Block
from utils.patch_embed import PatchEmbed

from utils.pos_embed import get_2d_sincos_pos_embed


# Main changes img_size adjusted to 12 channel ECG signal - 12*1000
# Functions - Patchify and unpatchify
# Other functions remains the same


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(12, 1000), patch_size=(1, 50), in_chans=1,
                 embed_dim=128, depth=6, num_heads=8,
                 decoder_embed_dim=64, decoder_depth=3, decoder_num_heads=8,
                 mlp_ratio=3., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.linear1 = nn.Linear(1, 32, bias=True)
        # self.linear2 = nn.Linear(1, 32, bias=True)
        self.output_shape = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Discriminator Specifics
        self.discriminate = nn.Linear(embed_dim, 1, bias = True)

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]* patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (12, self.patch_embed.num_patches//12), cls_token=True)
        # grid = (height, width). height = 12 here for 12 lead ecg signals
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],  (12, self.patch_embed.num_patches//12), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
                    
        for layer in self.patch_embed.patch:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                # if layer.bias is not None:
                #     torch.nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 1.0)
        

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W) - 12 channel ECG - H = No. of channels, W = Length of ECG signal (1000 in this case)
        x: (N, L, patch_size_height*patch_size_width*1)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph*pw * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size_height*patch_size_width*1)
        imgs: (N, 1, H, W) - 12 channel ECG - H = No. of channels, W = Length of ECG signal (1000 in this case)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        # h = w = int(x.shape[1]**.5)
        # assert h * w == x.shape[1]
        h = 12
        w = x.shape[1]//12
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # print("before patch embed = "+str(x.size()))
        x = self.patch_embed(x)
        # print("after patch embed = "+str(x.size()))
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print(x.size())

        return x, mask, ids_restore, 
    

    def discriminator(self, currupt_img):

        x = self.patch_embed(currupt_img)
        # print("after patch embed = "+str(x.size()))
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # x = x.view(x.size(0), -1)

        x = self.discriminate(x)
        return torch.sigmoid(x)
    

    def discriminator_loss(self, x, mask):
        # Real and fake discriminator outputs
        output = self.discriminator(x)
        output = output[:, 1:, 0]
        target = 1 - mask
        target = target.double()

        disc_loss = torch.nn.BCELoss()
        return disc_loss(output, target)
    

    def adv_loss(self, currupt_img, mask):
        target = 1 - mask  # This flips the mask values
        output = self.discriminator(currupt_img)
        disc_preds = output[:, 1:, 0]

        # Reshape target to match the discriminator output shape
        target = target.view(disc_preds.shape)
        target = target.float()

        # Calculate the number of correct predictions for original and reconstructed patches
        corr_orig = (torch.log(disc_preds + 1e-8) * target).sum()/(target.sum())
        corr_recons = (torch.log((1-disc_preds + 1e-8))*(1 - target)).sum()/((1-target).sum())
        # print(corr_orig)
        # print(corr_orig + corr_recons)
        return (corr_orig) + (corr_recons) 


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (N, 1, H, W) - 12 channel ECG - H = No. of channels, W = Length of ECG signal (1000 in this case)
        x: (N, L, patch_size_height*patch_size_width*1)
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # pred = self.unpatchify(pred)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        if torch.isnan(target).any():
            print("NaN values found in target")
        if torch.isnan(pred).any():
            print("NaN values found in pred tensors")

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # loss = (loss).sum() / len(loss)*240
        # print(loss)
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # print(imgs.size())
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # print(latent.size())
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # print(pred.size())
        mae_loss = self.forward_loss(imgs, pred, mask)

        # loss = loss + self.adaptive_weight()*self.adv_loss()
        ### 
        # currupt_img = reconstructed masked patches + unmasked patches
        img_patched = self.patchify(imgs)
        currupt_img = torch.zeros(img_patched.size())
        mask1 = mask.unsqueeze(-1).expand_as(pred)
        currupt_img = torch.where(mask1 == 1, pred, img_patched)
        currupt_img = self.unpatchify(currupt_img)
        ###

        disc_loss = self.discriminator_loss(currupt_img, mask)
        adv_loss = self.adv_loss(currupt_img, mask)

        return mae_loss, pred, mask, disc_loss, adv_loss, currupt_img

# Model architecture as described in the paper.
def mae_vit_1dcnn(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=(1, 50), embed_dim=128, depth=6, num_heads=8,
        decoder_embed_dim=64, decoder_depth=3, decoder_num_heads=8,
        mlp_ratio=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
