"""顔配置の事前学習を初期値とし、独立した奥行きCNNを学習する。"""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .face_features import FaceFeatures,face_image_tensor

POOL=4
CHANNELS=160
HIDDEN=128

class DepthNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=FaceFeatures('outputs/learning/separate-v2/openness','mps').model
        self.head=nn.Sequential(nn.Linear(CHANNELS*POOL*POOL,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,1))
    def forward(self,images):return self.head(F.adaptive_avg_pool2d(self.backbone.image_features(images),(POOL,POOL)).flatten(1)).flatten()

class NeuralPosition:
    def __init__(self,baseline,folder,sha):self.baseline=baseline;self.folder=str(folder);self.sha=sha;self.network=None
    def predict_image(self,features,content):
        if self.network is None:
            folder=Path(self.folder);path=folder/'weights.pt'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.sha:raise ValueError('奥行きCNN重みが不一致')
            self.network=DepthNetwork().to('mps').eval();self.network.load_state_dict(torch.load(path,map_location='mps',weights_only=True))
            self.normalization=json.loads((folder/'normalization.json').read_bytes())
        with torch.inference_mode():z=self.network(face_image_tensor(content)[None].to('mps')).item()
        point=self.baseline.predict_image(features,content).copy();point[2]=z*self.normalization['scale']+self.normalization['mean'];return point
