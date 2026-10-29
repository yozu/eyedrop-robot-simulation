"""画像内の点眼目標を、奥行き学習の補助教師として使う。"""
import torch
from torch.nn import functional as F
from .neural_depth import POOL

SIGMA = .025
AUXILIARY_WEIGHT = .01


def depth_and_maps(network, images):
    latent = network.backbone.image_features(images)
    depth = network.head(F.adaptive_avg_pool2d(latent, (POOL, POOL)).flatten(1)).flatten()
    return depth, network.backbone.readout(latent)[:, :1]


def target_coordinates(logits):
    height, width = logits.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device, dtype=logits.dtype),
                          torch.linspace(0, 1, width, device=logits.device, dtype=logits.dtype), indexing='ij')
    grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
    return logits[:, 0].flatten(1).softmax(-1) @ grid


def target_peak_coordinates(logits):
    if logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape[-2:]) < 3:
        raise ValueError('補間には1点の画像分布が必要です')
    maps = logits.detach().cpu().double()
    height, width = maps.shape[-2:]
    points = []
    for image in maps[:, 0]:
        if not torch.isfinite(image).all():
            raise ValueError('画像分布が有限ではありません')
        flat = int(image.argmax())
        y, x = divmod(flat, width)
        if x in (0, width - 1) or y in (0, height - 1):
            raise ValueError('分布の最大位置が画像端です')
        center = image[y, x]
        gradient = torch.stack(((image[y, x + 1] - image[y, x - 1]) / 2,
                                (image[y + 1, x] - image[y - 1, x]) / 2))
        cross = (image[y + 1, x + 1] - image[y + 1, x - 1]
                 - image[y - 1, x + 1] + image[y - 1, x - 1]) / 4
        hessian = torch.stack((torch.stack((image[y, x + 1] - 2 * center + image[y, x - 1], cross)),
                               torch.stack((cross, image[y + 1, x] - 2 * center + image[y - 1, x]))))
        if not (torch.linalg.eigvalsh(hessian) < 0).all():
            raise ValueError('分布の最大位置が明瞭ではありません')
        offset = -torch.linalg.solve(hessian, gradient)
        if not torch.isfinite(offset).all() or (offset.abs() > 1).any():
            raise ValueError('最大位置の補間が範囲外です')
        points.append(torch.stack(((x + offset[0]) / (width - 1), (y + offset[1]) / (height - 1))))
    return torch.stack(points).to(device=logits.device, dtype=logits.dtype)


def target_loss(logits, target, labelled):
    if logits.ndim != 4 or logits.shape[1] != 1 or target.shape != (len(logits), 2) or labelled.shape != (len(logits),):
        raise ValueError('点眼目標の教師と予測の形状が不正です')
    if not labelled.any():
        return logits.sum() * 0
    selected = target[labelled]
    if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
        raise ValueError('点眼目標の教師は画像内の有限座標が必要です')
    height, width = logits.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device, dtype=logits.dtype),
                          torch.linspace(0, 1, width, device=logits.device, dtype=logits.dtype), indexing='ij')
    grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
    distance = (grid[None] - selected[:, None]).square().sum(-1)
    distribution = (-distance / (2 * SIGMA ** 2)).softmax(-1)
    return -(distribution * logits[labelled, 0].flatten(1).log_softmax(-1)).sum(-1).mean()
