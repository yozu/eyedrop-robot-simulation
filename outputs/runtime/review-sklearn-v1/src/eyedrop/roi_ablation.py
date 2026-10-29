"""同じ画素密度での周辺除去と、切り出し解像度を別々に比較する。"""
import io
import numpy as np
from PIL import Image
from .tracked_roi import roi_box

IMAGE_SIDE=256

def vector(content,features,mode):
    box=roi_box(features)
    with Image.open(io.BytesIO(content)) as im:
        if im.size!=(IMAGE_SIDE,IMAGE_SIDE):raise ValueError('256角画像が必要')
        im=im.convert('RGB')
        if mode.startswith('crop'):
            side=int(mode[4:]);pixels=np.asarray(im.crop(tuple(box*IMAGE_SIDE)).resize((side,side),Image.Resampling.BILINEAR),dtype=np.float32)/255
        elif mode in ('full32','mask32'):
            pixels=np.asarray(im.resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255
            if mode=='mask32':
                y,x=np.mgrid[:32,:32];inside=(x/32>=box[0])&(x/32<=box[2])&(y/32>=box[1])&(y/32<=box[3]);pixels=pixels.copy();pixels[~inside]=.5
        else:raise ValueError('未知の表現')
    return np.r_[pixels.flatten(),(box[:2]+box[2:])/2,box[2:]-box[:2],features[:8]].astype(np.float32)

class AblationPosition:
    def __init__(self,baseline,regressor,mode):self.baseline=baseline;self.regressor=regressor;self.mode=mode
    def predict_image(self,features,content):
        point=self.baseline.predict_image(features,content).copy()
        point[2]=self.regressor.predict(vector(content,features,self.mode)[None])[0]
        return point
