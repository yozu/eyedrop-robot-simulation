"""画像と顔教師点を同時に変換する学習専用処理。推論には使用しない。"""
import torch
from torch.nn import functional as F

SCALE_MIN=.85
SCALE_MAX=1.15
SHIFT_MAX=.08
COLOR_MIN=.65
COLOR_MAX=1.35
GRAY_PROBABILITY=.3
LUMA_WEIGHTS=(.299,.587,.114)


def transform_face_batch(images,points,visible,scales,shifts):
    batch=images.shape[0]
    if scales.shape!=(batch,) or shifts.shape!=(batch,2) or (scales<=0).any():
        raise ValueError("画像変換の倍率・移動量が不正です")
    transformed=(points-.5)*scales[:,None,None]+.5+shifts[:,None,:]
    mask=visible & ((transformed>=0)&(transformed<=1)).all(-1)
    theta=torch.zeros(batch,2,3,device=images.device,dtype=images.dtype)
    theta[:,0,0]=theta[:,1,1]=1/scales
    theta[:,:,2]=-2*shifts/scales[:,None]
    grid=F.affine_grid(theta,images.shape,align_corners=True)
    output=F.grid_sample(images,grid,align_corners=True,padding_mode="border")
    return output,transformed,mask


def augment_face_batch(images,points,visible):
    batch=images.shape[0];device=images.device
    scales=SCALE_MIN+(SCALE_MAX-SCALE_MIN)*torch.rand(batch,device=device)
    shifts=(torch.rand(batch,2,device=device)*2-1)*SHIFT_MAX
    images,points,visible=transform_face_batch(images,points,visible,scales,shifts)
    weights=images.new_tensor(LUMA_WEIGHTS)[None,:,None,None]
    gray=(images*weights).sum(1,keepdim=True)
    use_gray=torch.rand(batch,1,1,1,device=device)<GRAY_PROBABILITY
    images=torch.where(use_gray,gray,images)
    gain=COLOR_MIN+(COLOR_MAX-COLOR_MIN)*torch.rand(batch,3,1,1,device=device)
    return (images*gain).clamp(0,1),points,visible
