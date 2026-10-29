"""RGBから推定した眼の近傍を、元の画素解像度で補正する。"""
import torch
from torch import nn
from torch.nn import functional as F

ROI_SIDE = 48
IMAGE_SIDE = 256
HALF_WIDTH = (ROI_SIDE - 1) / 2
SIGMA_PIXELS = 1.5
COORDINATE_WEIGHT = 1.


def crop_images(images, centers):
    if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIDE, IMAGE_SIDE) or centers.shape != (len(images), 2):
        raise ValueError('RGB画像または切り出し中心の形状が不正です')
    offset = (torch.arange(ROI_SIDE, device=images.device, dtype=images.dtype) - HALF_WIDTH) / IMAGE_SIDE
    y, x = torch.meshgrid(offset, offset, indexing='ij')
    grid = (centers[:, None, None] + torch.stack((x, y), -1)[None]) * 2 - 1
    if not torch.isfinite(grid).all() or ((grid < -1) | (grid > 1)).any():
        raise ValueError('眼の切り出し範囲が画像外です')
    return F.grid_sample(images, grid, mode='bilinear', align_corners=False)


def local_to_global(local, centers):
    return centers + (local - .5) * (ROI_SIDE - 1) / IMAGE_SIDE


def global_to_local(target, centers):
    return .5 + (target - centers) * IMAGE_SIDE / (ROI_SIDE - 1)


class LocalTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        y, x = torch.meshgrid(torch.linspace(0, 1, ROI_SIDE), torch.linspace(0, 1, ROI_SIDE), indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y)), persistent=False)

    def forward(self, images):
        maps = self.layers(torch.cat((images, self.coordinates[None].expand(len(images), -1, -1, -1) * 2 - 1), 1))
        probability = maps.flatten(1).softmax(-1)
        point = probability @ self.coordinates.flatten(1).T
        return point, maps

    def loss(self, points, maps, target):
        if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
            raise ValueError('切り出し範囲に教師の眼がありません')
        grid = self.coordinates.flatten(1).T
        distance = (grid[None] - target[:, None]).square().sum(-1)
        distribution = (-distance / (2 * (SIGMA_PIXELS / (ROI_SIDE - 1)) ** 2)).softmax(-1)
        location = -(distribution * maps.flatten(1).log_softmax(-1)).sum(-1).mean()
        return location + COORDINATE_WEIGHT * F.smooth_l1_loss(points, target)
