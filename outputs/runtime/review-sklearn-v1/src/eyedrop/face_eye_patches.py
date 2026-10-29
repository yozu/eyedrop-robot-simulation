"""予測した眼位置のRGB領域から開眼量を学ぶ診断モデル。"""
import torch
from torch import nn
from torch.nn import functional as F
from .face_layout import FaceLayoutEncoder, spatial_coordinates, IMAGE_SIDE

PATCH_SIDE=32
EYE_COUNT=2
PATCH_WIDTH_RATIO=.8
PATCH_MIN_WIDTH=.06
PATCH_MAX_WIDTH=.25
PATCH_HEIGHT_RATIO=.65
LOCAL_EYE_VERSION="face-local-eye-v1"
NORMALIZED_EYE_VERSION="face-local-eye-normalized-v1"
CONTRAST_EPSILON=.01


def normalized_eye_patches(patches):
    # 色・照明の違いより、まぶたと眼の形を学習させる。
    gray=patches.mean(dim=-3,keepdim=True)
    mean=gray.mean(dim=(-2,-1),keepdim=True)
    scale=gray.std(dim=(-2,-1),keepdim=True,unbiased=False).clamp_min(CONTRAST_EPSILON)
    return ((gray-mean)/scale).expand_as(patches)


def eye_patches(image,coordinates):
    centers=coordinates[:,:EYE_COUNT]
    width=((centers[:,0]-centers[:,1]).norm(dim=-1)*PATCH_WIDTH_RATIO).clamp(PATCH_MIN_WIDTH,PATCH_MAX_WIDTH)
    axis=torch.linspace(-.5,.5,PATCH_SIDE,device=image.device,dtype=image.dtype)
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    offsets=torch.stack((x,y*PATCH_HEIGHT_RATIO),dim=-1)
    grid=centers[:,:,None,None,:]+offsets[None,None]*width[:,None,None,None,None]
    expanded=image[:,None].expand(-1,EYE_COUNT,-1,-1,-1).flatten(0,1)
    patches=F.grid_sample(expanded,(grid*2-1).flatten(0,1),align_corners=True,padding_mode="border")
    return patches.unflatten(0,(image.shape[0],EYE_COUNT))


class LocalEyePerceptionEncoder(FaceLayoutEncoder):
    def __init__(self,jitter_pixels=0.):
        super().__init__()
        if not 0<=jitter_pixels<=PATCH_SIDE/2:
            raise ValueError("眼切り出しのずれ量が範囲外です")
        self.jitter_pixels=jitter_pixels
        self.openness_head=nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(32*8*8,64),nn.ReLU(),nn.Linear(64,1))

    def perceive(self,image):
        maps=self(image)
        # 開眼損失で眼位置が移動しないよう、位置側の勾配を切る。
        coordinates=spatial_coordinates(maps).detach()
        if self.training and self.jitter_pixels:
            coordinates=coordinates.clone()
            coordinates[:,:EYE_COUNT]+=(torch.rand_like(coordinates[:,:EYE_COUNT])*2-1)*self.jitter_pixels/(IMAGE_SIDE-1)
        patches=eye_patches(image,coordinates)
        logits=self.openness_head(patches.flatten(0,1)).reshape(image.shape[0],EYE_COUNT)
        return maps,logits.mean(dim=1).sigmoid()


class NormalizedEyePerceptionEncoder(LocalEyePerceptionEncoder):
    def perceive(self,image):
        maps=self(image)
        coordinates=spatial_coordinates(maps).detach()
        if self.training and self.jitter_pixels:
            coordinates=coordinates.clone()
            coordinates[:,:EYE_COUNT]+=(torch.rand_like(coordinates[:,:EYE_COUNT])*2-1)*self.jitter_pixels/(IMAGE_SIDE-1)
        patches=normalized_eye_patches(eye_patches(image,coordinates))
        logits=self.openness_head(patches.flatten(0,1)).reshape(image.shape[0],EYE_COUNT)
        return maps,logits.mean(dim=1).sigmoid()
