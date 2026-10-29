"""平行な左右カメラの同じ高さという制約で画像分布を結び付ける。"""
import torch


def epipolar_coordinates(logits):
    if logits.ndim!=5 or logits.shape[1:3]!=(2,1) or min(logits.shape[-2:])<2 or not torch.isfinite(logits).all():
        raise ValueError('有限な左右1点の画像分布が必要です')
    maps=logits[:,:,0];height,width=maps.shape[-2:]
    row_evidence=maps.logsumexp(-1)
    shared_row=row_evidence.sum(1).softmax(-1)
    x=torch.linspace(0,1,width,device=logits.device,dtype=logits.dtype)
    y=torch.linspace(0,1,height,device=logits.device,dtype=logits.dtype)
    horizontal=((maps.softmax(-1)*x).sum(-1)*shared_row[:,None,:]).sum(-1)
    vertical=(shared_row*y).sum(-1)[:,None].expand(-1,2)
    return torch.stack((horizontal,vertical),dim=-1)
