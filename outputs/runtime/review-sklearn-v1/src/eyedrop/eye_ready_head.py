"""画像から予測した眼領域だけで、診断候補の開眼準備判定を行う。"""
import numpy as np
import torch
from .face_features import face_image_tensor,FEATURE_COUNT
from .face_eye_patches import eye_patches,normalized_eye_patches,EYE_COUNT

READY_PROBABILITY=.5


def ready_from_image(head,content,features,device):
    features=np.asarray(features)
    if features.shape!=(FEATURE_COUNT,) or not np.isfinite(features).all():raise ValueError('予測顔配置が不正です')
    coords=torch.tensor((features[:FEATURE_COUNT-1].reshape(-1,2)+1)/2,dtype=torch.float32)[None]
    with torch.inference_mode():
        patches=normalized_eye_patches(eye_patches(face_image_tensor(content)[None],coords)).to(device)
        probability=float(head(patches.flatten(0,1)).reshape(EYE_COUNT).mean().sigmoid().cpu())
    if not np.isfinite(probability):raise ValueError('開眼準備確率が非有限です')
    return float(probability>=READY_PROBABILITY)
