# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

from functools import partial

import numpy as np
import timm.models
import torch
import torch.nn as nn
from einops import rearrange

from timm.models.vision_transformer import PatchEmbed, Block, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 asymmetric_decoder=False, mask_ratio=0.75, vis_mask_ratio=0.,
                 learning_loss=True):
        super().__init__()

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.learning_loss = learning_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics (projector, predictor, and loss predictor)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # reconstructor (e.g., projector at feature-level)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)          # dino/ema
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)  # clip

        # loss predictor
        if self.learning_loss:
            self.decoder_blocks_losspred = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                      norm_layer=norm_layer)
                for i in range(decoder_depth)])
            self.decoder_norm_losspred = norm_layer(decoder_embed_dim)
            self.decoder_pred_losspred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, 'vis_mask_token'):
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        # x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, _, D = x.shape
        x = x[~mask].reshape(N, -1, D)

        if self.vis_mask_ratio > 0:
            vis_mask_token = self.vis_mask_token + self.pos_embed[:, 1:, :]
            vis_mask_token = vis_mask_token.expand(N, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(N, -1, D)
            L = x.size(1)
            noise = torch.rand(N, L, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(L * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones([N, L], device=x.device)
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(vis_mask, dim=1, index=ids_restore).unsqueeze(-1)

            x = x * (1. - vis_mask) + vis_mask_token * vis_mask

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        x_vis = x[:, 1:, :]
        N, _, D = x_vis.shape

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        pos_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_mask = expand_pos_embed[mask].reshape(N, -1, D)

        x_ = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)

        # add cls_token + decoder_pos_embed
        x = torch.cat([x[:, :1, :] + self.decoder_pos_embed[:, :1, :], x_], dim=1)
        loss_pred = x.clone()

        # apply reconstructor
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        if self.learning_loss:
            # apply loss predictor
            for blk in self.decoder_blocks_losspred:
                loss_pred = blk(loss_pred)
            loss_pred = self.decoder_norm_losspred(loss_pred)
            loss_pred = self.decoder_pred_losspred(loss_pred)
            loss_pred = loss_pred[:, 1:, :]  # (N, L, 1)

            return x, pos_mask.shape[1], loss_pred.mean(dim=-1)

        return x, pos_mask.shape[1]

    def forward_loss(self, pred, target, mask):
        """
        pred: [N, mask, D]
        target: [N, L, D]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        loss = ((pred - target) ** 2).sum(dim=-1)

        return {'mean': loss.mean(), 'matrix': loss}

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask)  # returned mask may change

        if self.learning_loss:
            pred, mask_num, loss_pred = self.forward_decoder(latent, mask)  # [N, L, p*p*3]
        else:
            pred, mask_num = self.forward_decoder(latent, mask)
        # loss = self.forward_loss(imgs, pred[:, -mask_num:], mask)
        # return loss, pred, mask
        out = {
            'pix_pred': pred,
            'mask': mask,
            'mask_num': mask_num,
            'features': latent,
        }

        if self.learning_loss:
            out['loss_pred'] = loss_pred

        return out

    def generate_mask(self, loss_pred, mask_ratio=0.75, images=None,  guide=True, epoch=0, total_epoch=200):
        loss_pred = loss_pred.squeeze()
        N, L = loss_pred.shape
        len_keep = int(L * (1 - mask_ratio))

        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)  # (N, L)

        # keep `keep_ratio` loss and `1 - keep_ratio` random
        keep_ratio = 0.2
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()

        if guide:
            keep_ratio = float((epoch + 1) / total_epoch) * 0.5

        ## top 0 -> 0.5
        if int((L - len_keep) * keep_ratio) <= 0:
            # random
            noise = torch.randn(N, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                ## mask top `keep_ratio` loss and `1 - keep_ratio` random
                len_loss = int((L - len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=loss_pred.device)
        mask[:, :len_keep] = 0
        # unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_learning_loss(self, loss_pred, mask, loss_target, relative=False):
        """
        loss_pred: [N, L, 1]
        mask: [N, L], 0 is keep, 1 is remove,
        loss_target: [N, L]
        """
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)
        assert self.learning_loss

        if relative:
            # binary classification for LxL
            labels_positive = loss_target.unsqueeze(1) > loss_target.unsqueeze(2)
            labels_negative = loss_target.unsqueeze(1) < loss_target.unsqueeze(2)
            labels_valid = labels_positive + labels_negative

            loss_matrix = loss_pred.unsqueeze(1) - loss_pred.unsqueeze(2)
            loss = - labels_positive.int() * torch.log(torch.sigmoid(loss_matrix) + 1e-6) \
                   - labels_negative.int() * torch.log(1 - torch.sigmoid(loss_matrix) + 1e-6)

            return loss.sum() / labels_valid.sum()

        else:
            # normalize by each image
            mean = loss_target.mean(dim=1, keepdim=True)
            var = loss_target.var(dim=1, keepdim=True)
            loss_target = (loss_target - mean) / (var + 1.e-6) ** .5  # [N, L, 1]

            loss = (loss_pred - loss_target) ** 2
            loss = loss.mean()
            return loss


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