"""カメラ画像から予測した顔配置を操作方策の特徴へ変換する。"""
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .face_layout import FacePerceptionEncoder,PERCEPTION_VERSION,spatial_coordinates,PARTS,IMAGE_SIDE
from .model_integrity import verified_model_bytes

from .face_eye_patches import (LocalEyePerceptionEncoder,LOCAL_EYE_VERSION,
    NormalizedEyePerceptionEncoder,NORMALIZED_EYE_VERSION)

FEATURE_COUNT=PARTS*2+1
CACHE_VERSION='predicted-face-features-v1'


def face_image_tensor(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.mode=='RGBA' and image.getchannel('A').getextrema()==(255,255):image=image.convert('RGB')
        if image.mode!='RGB' or image.size!=(IMAGE_SIDE,IMAGE_SIDE):raise ValueError('256角のRGB画像が必要です')
        array=np.array(image)
    return torch.from_numpy(array).permute(2,0,1).float()/255.


def pack_features(maps,openness):
    if openness.shape!=(maps.shape[0],):raise ValueError('開眼予測の形状が不正です')
    return torch.cat((spatial_coordinates(maps).flatten(1),openness[:,None]),dim=1)*2-1


class FaceFeatures:
    def __init__(self,folder,device,*,expected_sha256=None):
        folder=Path(folder);report=json.loads((folder/'report.json').read_bytes())
        if report['architecture']==PERCEPTION_VERSION:model_class=FacePerceptionEncoder
        elif report['architecture']==LOCAL_EYE_VERSION:model_class=LocalEyePerceptionEncoder
        elif report['architecture']==NORMALIZED_EYE_VERSION:model_class=NormalizedEyePerceptionEncoder
        else:raise ValueError('顔予測モデルの構成が異なります')
        self.sha256=report['model_sha256']
        if expected_sha256 is not None and self.sha256!=expected_sha256:
            raise ValueError('固定した顔予測モデルのハッシュと一致しません')
        content=verified_model_bytes(folder/'weights.pt',self.sha256)
        self.device=device
        self.model=model_class().to(device).eval()
        self.model.load_state_dict(torch.load(io.BytesIO(content),map_location=device,weights_only=True))

    def batch(self,images):
        with torch.inference_mode():
            maps,openness=self.model.perceive(images.to(self.device))
            return pack_features(maps,openness)

    def extract(self,content):
        return self.batch(face_image_tensor(content).unsqueeze(0))[0]
