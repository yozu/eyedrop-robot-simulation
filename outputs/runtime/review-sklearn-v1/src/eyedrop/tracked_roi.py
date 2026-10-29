"""予測した顔パーツに追従し、元画像の位置と縮尺も残す。"""
import io
import numpy as np
from PIL import Image

SIDE=256
PATCH_SIDE=16
MIN_SPAN=.04
MARGIN=.6

def roi_box(features):
    points=(np.asarray(features[:8],dtype=float).reshape(4,2)+1)/2
    if not np.isfinite(points).all():raise ValueError('顔座標が非有限')
    low=points.min(0);high=points.max(0);span=np.maximum(high-low,MIN_SPAN)
    return np.r_[np.maximum(0,low-MARGIN*span),np.minimum(1,high+MARGIN*span)]

def representation(content,features,geometry=True):
    box=roi_box(features)
    with Image.open(io.BytesIO(content)) as image:
        if image.size!=(SIDE,SIDE):raise ValueError('256角画像が必要')
        patch=image.convert('RGB').crop(tuple(box*SIDE)).resize((PATCH_SIDE,PATCH_SIDE),Image.Resampling.BILINEAR)
        pixels=np.asarray(patch,dtype=np.float32).reshape(-1)/255
    if not geometry:return pixels
    center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    return np.r_[pixels,center,size,np.asarray(features[:8])].astype(np.float32)

class TrackedPosition:
    def __init__(self,baseline,regressor,geometry):
        self.baseline=baseline;self.regressor=regressor;self.geometry=geometry
    def predict_image(self,features,content):
        point=self.baseline.predict_image(features,content).copy()
        point[2]=self.regressor.predict(representation(content,features,self.geometry)[None])[0]
        return point
