"""画像から顔パーツ配置を学ぶ補助モデル。教師座標は損失計算だけに使用する。"""
import torch
from torch import nn
from torch.nn import functional as F

LANDMARKS=("left_eye", "right_eye", "nose_tip", "mouth_center")
PARTS=len(LANDMARKS)
IMAGE_SIDE=256
HEATMAP_SIGMA=.025
COORDINATE_WEIGHT=1.
RELATION_WEIGHT=1.


def spatial_coordinates(logits):
    if logits.ndim!=4 or logits.shape[1]!=PARTS:
        raise ValueError("four landmark heatmaps required")
    height,width=logits.shape[-2:]
    if min(height,width)<2:
        raise ValueError("heatmaps must have spatial extent")
    y,x=torch.meshgrid(torch.linspace(0,1,height,device=logits.device,dtype=logits.dtype),
                       torch.linspace(0,1,width,device=logits.device,dtype=logits.dtype),indexing="ij")
    grid=torch.stack((x.flatten(),y.flatten()),dim=-1)
    return logits.flatten(2).softmax(-1)@grid


def layout_loss(logits,targets,visible):
    predicted=spatial_coordinates(logits)
    if targets.shape!=predicted.shape or visible.shape!=predicted.shape[:2] or visible.dtype!=torch.bool:
        raise ValueError("landmark label shapes or mask invalid")
    if not visible.any():
        raise ValueError("visible landmarks required")
    if not torch.isfinite(targets[visible]).all() or ((targets[visible]<0)|(targets[visible]>1)).any():
        raise ValueError("visible coordinates must be finite and normalized")
    # 遮蔽点の未定義座標が演算へ混入しないよう、計算前に除外する。
    safe=torch.where(visible.unsqueeze(-1),targets,torch.zeros_like(targets))
    height,width=logits.shape[-2:]
    y,x=torch.meshgrid(torch.linspace(0,1,height,device=logits.device,dtype=logits.dtype),
                       torch.linspace(0,1,width,device=logits.device,dtype=logits.dtype),indexing="ij")
    grid=torch.stack((x.flatten(),y.flatten()),dim=-1)
    distance=(grid[None,None]-safe.unsqueeze(2)).square().sum(-1)
    distribution=(-distance/(2*HEATMAP_SIGMA**2)).softmax(-1)
    spatial=-(distribution*logits.flatten(2).log_softmax(-1)).sum(-1)[visible].mean()
    coordinate=F.smooth_l1_loss(predicted[visible],safe[visible])
    # 点ごとの誤差に加え、左右の目・鼻・口の相対配置を同時に学習する。
    pairs=torch.triu(torch.ones(PARTS,PARTS,dtype=torch.bool,device=logits.device),diagonal=1)
    pair_mask=visible[:,:,None]&visible[:,None,:]&pairs
    prediction_delta=predicted[:,:,None,:]-predicted[:,None,:,:]
    target_delta=safe[:,:,None,:]-safe[:,None,:,:]
    relation=F.smooth_l1_loss(prediction_delta[pair_mask],target_delta[pair_mask]) if pair_mask.any() else logits.sum()*0
    return spatial+COORDINATE_WEIGHT*coordinate+RELATION_WEIGHT*relation


ARCHITECTURE_VERSION="face-layout-context-v2"


class FaceLayoutEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine=nn.Sequential(
            nn.Conv2d(3,16,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU())
        self.context=nn.Sequential(
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(128,128,3,dilation=2,padding=2),nn.ReLU(),
            nn.Conv2d(128,128,3,dilation=2,padding=2),nn.ReLU())
        self.readout=nn.Sequential(nn.Conv2d(160,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,PARTS,1))

    def image_features(self,image):
        if image.ndim!=4 or image.shape[1:]!=(3,IMAGE_SIDE,IMAGE_SIDE):
            raise ValueError("256x256 RGB batch required")
        fine=self.fine(image)
        context=F.interpolate(self.context(fine),size=fine.shape[-2:],mode="bilinear",align_corners=False)
        return torch.cat((fine,context),dim=1)

    def forward(self,image):
        return self.readout(self.image_features(image))


PERCEPTION_VERSION="face-layout-openness-v3"
EYE_INDICES=(0,1)
FEATURE_CHANNELS=160
OPENNESS_HIDDEN=64


class FacePerceptionEncoder(FaceLayoutEncoder):
    def __init__(self):
        super().__init__()
        self.openness_head=nn.Sequential(
            nn.Linear(FEATURE_CHANNELS*len(EYE_INDICES),OPENNESS_HIDDEN),nn.ReLU(),
            nn.Linear(OPENNESS_HIDDEN,1))

    def perceive(self,image):
        features=self.image_features(image)
        maps=self.readout(features)
        # 画像から予測した左右の目の分布で特徴を集める。教師の眼位置は渡さない。
        attention=maps[:,EYE_INDICES].flatten(2).softmax(-1)
        eyes=torch.bmm(attention,features.flatten(2).transpose(1,2)).flatten(1)
        openness=self.openness_head(eyes).squeeze(-1).sigmoid()
        return maps,openness
