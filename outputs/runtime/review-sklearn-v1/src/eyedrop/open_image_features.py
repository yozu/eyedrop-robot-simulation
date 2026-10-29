"""画像から推定した眼位置で切り出した領域を、開眼専用の特徴へ変換する。"""
import numpy as np
import torch
from torch.nn import functional as F
from .face_features import face_image_tensor,FEATURE_COUNT
from .face_eye_patches import eye_patches,normalized_eye_patches,PATCH_SIDE,EYE_COUNT

POOL=2
INPUT_COUNT=FEATURE_COUNT+EYE_COUNT*(PATCH_SIDE//POOL)**2


def eye_image_features(content,face):
    face=np.asarray(face,dtype=float)
    if face.shape!=(FEATURE_COUNT,) or not np.isfinite(face).all():raise ValueError('顔特徴が不正です')
    coordinates=torch.tensor((face[:FEATURE_COUNT-1].reshape(-1,2)+1)/2,dtype=torch.float32)[None]
    with torch.inference_mode():
        patches=normalized_eye_patches(eye_patches(face_image_tensor(content)[None],coordinates))
        gray=patches[0,:,0:1]
        pixels=F.avg_pool2d(gray,POOL).flatten().numpy()
    return np.r_[face,pixels]
