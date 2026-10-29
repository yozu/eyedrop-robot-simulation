"""固定した顔認識CNNの空間特徴から奥行きを読む。"""
import numpy as np
import torch
from torch.nn import functional as F
from .face_features import FaceFeatures,face_image_tensor

POOL=4
MODEL_FOLDER='outputs/learning/separate-v2/openness'

def encode(encoder,images):
    with torch.inference_mode():
        f=encoder.model.image_features(images.to(encoder.device))
        maps=encoder.model.readout(f)
        attention=maps.flatten(2).softmax(-1)
        parts=torch.bmm(attention,f.flatten(2).transpose(1,2)).flatten(1)
        spatial=F.adaptive_avg_pool2d(f,(POOL,POOL)).flatten(1)
        return torch.cat((spatial,parts),dim=1).cpu().numpy()

class SemanticPosition:
    def __init__(self,baseline,regressor):self.baseline=baseline;self.regressor=regressor;self.encoder=None
    def predict_image(self,features,content):
        if self.encoder is None:self.encoder=FaceFeatures(MODEL_FOLDER,'mps')
        x=encode(self.encoder,face_image_tensor(content)[None])
        point=self.baseline.predict_image(features,content).copy();point[2]=self.regressor.predict(x)[0]
        return point
