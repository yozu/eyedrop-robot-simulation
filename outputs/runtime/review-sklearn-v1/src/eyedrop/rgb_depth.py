"""RGB全体の輪郭と縮尺から、世界座標の奥行きを回帰する。"""
import io
import numpy as np
from PIL import Image

IMAGE_SIDE=32
DEPTH_AXIS=2

def image_vector(content,side=IMAGE_SIDE):
    with Image.open(io.BytesIO(content)) as image:
        if image.size!=(256,256):raise ValueError('256角の画像が必要です')
        if image.mode=='RGBA' and image.getchannel('A').getextrema()==(255,255):image=image.convert('RGB')
        if image.mode!='RGB':raise ValueError('不透明なRGB画像が必要です')
        return np.asarray(image.resize((side,side),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255.

class RGBDepthPosition:
    def __init__(self,baseline,regressor):
        self.baseline=baseline;self.regressor=regressor
    def predict_image(self,features,content):
        point=self.baseline.predict(np.asarray(features).reshape(1,-1))[0].copy()
        point[DEPTH_AXIS]=self.regressor.predict(image_vector(content).reshape(1,-1))[0]
        return point
