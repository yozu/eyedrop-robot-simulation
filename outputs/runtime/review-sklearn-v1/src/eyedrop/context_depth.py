"""ROI内側で境界を滑らかにし、除去領域を広げずに比較する。"""
import io
import numpy as np
from PIL import Image
from .tracked_roi import roi_box

GRID=32
EDGE_WIDTH=2.

def features(content,landmarks,mode):
    box=roi_box(landmarks);y,x=np.mgrid[:GRID,:GRID];x=x/GRID;y=y/GRID
    distances=np.minimum.reduce([x-box[0],box[2]-x,y-box[1],box[3]-y])*GRID
    inside=distances>=0
    if mode=='hard':weight=inside.astype(float)
    elif mode=='feather':weight=np.clip(distances/EDGE_WIDTH,0,1)
    elif mode=='full':weight=np.ones_like(x)
    elif mode=='outside':weight=~inside
    else:raise ValueError('未知の周辺比較')
    with Image.open(io.BytesIO(content)) as im:
        if im.size!=(256,256):raise ValueError('256角画像が必要')
        pixels=np.asarray(im.convert('RGB').resize((GRID,GRID),Image.Resampling.BILINEAR),dtype=np.float32)/255
    pixels=pixels*weight[:,:,None]+.5*(1-weight[:,:,None])
    return np.r_[pixels.flatten(),(box[:2]+box[2:])/2,box[2:]-box[:2],landmarks[:8]].astype(np.float32)

class ContextPosition:
    def __init__(self,baseline,regressor,mode):self.baseline=baseline;self.regressor=regressor;self.mode=mode
    def predict_image(self,landmarks,content):
        point=self.baseline.predict_image(landmarks,content).copy()
        point[2]=self.regressor.predict(features(content,landmarks,self.mode)[None])[0]
        return point
