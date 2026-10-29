"""RGBで粗く求めた目標の周辺を、余白補正なしで拡大する。"""
import numpy as np

IMAGE_SIDE=512
ROI_SIDE=128


def roi_box(predicted_uv):
    point=np.asarray(predicted_uv,dtype=float)
    if point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('画像予測の中心が不正です')
    left,top=np.rint(point*IMAGE_SIDE-ROI_SIDE/2).astype(int)
    if min(left,top)<0 or max(left,top)+ROI_SIDE>IMAGE_SIDE:raise ValueError('画像予測の領域が画角外です')
    return int(left),int(top),int(left+ROI_SIDE),int(top+ROI_SIDE)


def to_roi_coordinates(points,box):
    points=np.asarray(points,dtype=float)
    if points.shape[-1:]!=(2,) or not np.isfinite(points).all():raise ValueError('画像座標が不正です')
    local=(points*IMAGE_SIDE-np.array(box[:2]))/ROI_SIDE
    if ((local<0)|(local>1)).any():raise ValueError('教師点が画像予測による領域外です')
    return local


def from_roi_coordinates(points,box):
    points=np.asarray(points,dtype=float)
    if points.shape[-1:]!=(2,) or not np.isfinite(points).all() or ((points<0)|(points>1)).any():raise ValueError('領域内座標が不正です')
    return (points*ROI_SIDE+np.array(box[:2]))/IMAGE_SIDE
