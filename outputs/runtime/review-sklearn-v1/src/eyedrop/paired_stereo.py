"""予測した左右対応点を固定カメラの幾何へ接続する。正解座標は不要。"""
import numpy as np
from .camera_position import CAMERA,project,basis
from .stereo_geometry import BASELINE,FOCAL,SIDE,MIN_CAMERA_DEPTH,MAX_CAMERA_DEPTH,disparity_point

IMAGE_SIDE=512
MAX_EPIPOLAR_PIXELS=1.


def project_pair(point):
    point=np.asarray(point,dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('教師点が不正です')
    depth=float((point-CAMERA)@basis()[0])
    if depth<=0:raise ValueError('教師点がカメラの後方です')
    left=(project(point)+1)/2
    right=left-np.array([FOCAL/SIDE*BASELINE/depth,0.])
    pair=np.array([left,right])
    if np.any(pair<0) or np.any(pair>1):raise ValueError('教師点が画角外です')
    return pair


def triangulate_pair(pair):
    pair=np.asarray(pair,dtype=float)
    if pair.shape!=(2,2) or not np.isfinite(pair).all() or np.any(pair<0) or np.any(pair>1):
        raise ValueError('左右対応点が不正です')
    epipolar=abs(pair[0,1]-pair[1,1])*IMAGE_SIDE
    if epipolar>MAX_EPIPOLAR_PIXELS:raise ValueError('左右対応点の縦ずれが上限を超えました')
    disparity=(pair[0,0]-pair[1,0])*IMAGE_SIDE
    point=disparity_point(pair[0]*2-1,disparity,image_side=IMAGE_SIDE)
    depth=float((point-CAMERA)@basis()[0])
    if not MIN_CAMERA_DEPTH<=depth<=MAX_CAMERA_DEPTH:raise ValueError('視差からの距離が固定カメラの検証範囲外です')
    return {'eye_scene':point.tolist(),'disparity_pixels':float(disparity),'epipolar_pixels':float(epipolar)}
