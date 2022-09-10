import torch
import torch.nn as nn
from functools import partial


from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
import torch.nn.functional as F
__all__ = [
    'deit_tscam_tiny_patch16_224', 'deit_tscam_small_patch16_224', 'deit_tscam_base_patch16_224',
]


def tilling(features, ps=224):
    _, _, h, w = features.shape
    patches = []
    for splitted_features in torch.split(features, 224, dim=2):
        for patch in torch.split(splitted_features, 224, dim=3):
            patches.append(patch)
    return torch.cat(patches, dim=0), h // 224, w // 224

def merging(features, num_h, num_w, bs):
    features_list = list(torch.split(features, bs))
    index = 0
    ext_h_list = []
    for _ in range(num_h):
        ext_w_list = []
        for _ in range(num_w):
            ext_w_list.append(features_list[index])
            index += 1
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features

class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

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
        x = self.forward_blocks(x, attn_weights)
        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward_blocks(self, x, attn_weights):
        for blk in self.blocks:
            x, weights = blk(x)  # 8 197 384
            attn_weights.append(weights)
        return x

    def forward(self, x, return_cam=False, is_split=False):
        ###############################################
        # overlap cutting start
        ###############################################
        if is_split:
            B = x.shape[0]
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=True)
            x, num_h, num_w = tilling(x)
        ###############################################
        # overlap cutting end
        ###############################################
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if self.training:
            ###############################################
            # overlap merging start
            ###############################################
            if is_split:
                x_patch = merging(x_patch,num_h,num_w,B)
                x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)
            ###############################################
            # overlap merging end
            ###############################################
            return x_logits, x_patch
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
            ###############################################
            # overlap merging start
            ###############################################
            if is_split:
                cams = merging(cams,num_h,num_w,B)
                cams = F.interpolate(cams, size=(14, 14), mode='bilinear', align_corners=True)
                cams = F.relu(cams)
                x_patch = merging(x_patch,num_h,num_w,B)
                x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)
            ###############################################
            # overlap merging end
            ###############################################
            cams = F.relu(cams)
            return x_logits, cams


@register_model
def deit_tscam_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
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

@register_model
def deit_tscam_small_patch16_224(pretrained=False, **kwargs):
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


@register_model
def deit_tscam_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
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





