import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

__all__ = [
    'deit_tscam_tiny_patch16_224', 'deit_tscam_small_patch16_224', 'deit_tscam_base_patch16_224',
]


def dilation(x):
    _, _, h, w = x.shape
    H = max((h // 112 + 1) * 112, 224)
    W = max((w // 112 + 1) * 112, 224)
    padding = torch.nn.ZeroPad2d((0, W - w, 0, H - h))
    return padding(x)

def tilling(features, ps=224):
    _, _, h, w = features.shape
    assert h % (ps//2) == 0 and w % (ps//2) == 0, "son of biscuit"
    num_h = (h-ps) // (ps//2) + 1
    num_w = (w-ps) // (ps//2) + 1
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            patches.append(features[:,:,i*ps//2:(i+2)*ps//2,j*ps//2:(j+2)*ps//2])
    return torch.cat(patches, dim=0), num_h, num_w

def merging(features, num_h, num_w, bs):
    features_list = list(torch.split(features, bs))
    n, c, ps, ps = features_list[0].shape
    merge_feature = torch.zeros(n,c,ps*(num_h+1)//2,ps*(num_w+1)//2).cuda()
    mask = torch.zeros_like(merge_feature).cuda()
    for i in range(num_h):
        for j in range(num_w):
            merge_feature[:,:,i*ps//2:(i+2)*ps//2,j*ps//2:(j+2)*ps//2] += features_list[i*num_w+j]
            mask[:,:,i*ps//2:(i+2)*ps//2,j*ps//2:(j+2)*ps//2] += torch.ones_like(features_list[i*num_w+j])
    return merge_feature / mask

class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

    def forward_blocks(self, attn_weights, x):
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
        return x

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_weights = []
        n = 3
        for i in range(n):
            x = x + self.forward_blocks(attn_weights, x)
        weights = []
        for i in range(12):
            temp = torch.zeros_like(attn_weights[0])
            for j in range(n):
                temp = temp + attn_weights[i+12*j]
            weights.append(temp)
        attn_weights = weights
        # attn_weights = [attn_weights[i]+attn_weights[i+12]+attn_weights[i+24] for i in range(12)]
        """
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
        """
        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, return_cam=False):
        if return_cam:
            _, _, h1, w1 = x.shape
            x = dilation(x)
            _, _, h2, w2 = x.shape
        B = x.shape[0]
        x, num_h, num_w = tilling(x)
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        # x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if self.training:
            x_patch = merging(x_patch, num_h, num_w, B)
            return x_patch
        else:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            #cams = attn_weights.mean(0).mean(1)[:, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights[:-1].sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights.sum(0)[:][:, 1:, 1:].sum(1).reshape([n, h, w]).unsqueeze(1)
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            cams = cams * feature_map                           # B * C * 14 * 14

            cams = merging(cams, num_h, num_w, 2)
            _, _, h, w = cams.shape
            cams = cams[:, :, :math.ceil(h * h1 / h2), :math.ceil(w * w1 / w2)]
            cams = F.relu(cams)
            cams = cams[0] + cams[1].flip(-1)

            return cams

    def trainable_parameters(self):
        parameters_list = list(self.parameters())
        return (parameters_list[0:-2], parameters_list[-2:])


@register_model
def Net(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model