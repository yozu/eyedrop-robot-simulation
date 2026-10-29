"""保存済みの認識モデルをPandaの左右RGBへ接続する。"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .face_features import FaceFeatures, face_image_tensor
from .open_image_features import eye_image_features
from .stereo_control import StereoPosition
from .stereo_geometry import match_disparity, disparity_point
from .target_supervision import target_peak_coordinates
from .canonical_eye_head import read_canonical_regressor
from .semantic_depth import encode
from .model_integrity import FIXED_RGB_MODELS as EXPECTED_MODELS, verified_model_bytes, restore_sklearn_model

IMAGE_SIDE = 512
FACE_SIDE = 256
STEREO_PATCH_RADIUS = 8
LOCATOR_ADAPTATION_UPDATES = 300
CLOSED_POSITION_OPEN_THRESHOLD = .95
CANONICAL_MM_PER_M = 1000


def load_locator_candidate(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_bytes())
    if (report.get('initial_weights_sha256') != EXPECTED_MODELS['target-readout-v1/weights.pt']
            or report.get('training_updates') != LOCATOR_ADAPTATION_UPDATES
            or report.get('diagnostic_only') is not True):
        raise ValueError('画像位置の診断候補が固定した追加学習条件と異なります')
    candidate = FaceFeatures(folder, 'mps', expected_sha256=report.get('model_sha256'))
    if candidate.sha256 != report.get('model_sha256'):
        raise ValueError('画像位置の診断候補のハッシュが異なります')
    return candidate


def face_rgb(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.mode == 'RGBA' and image.getchannel('A').getextrema() == (255, 255):
            image = image.convert('RGB')
        if image.mode != 'RGB' or image.size != (IMAGE_SIDE, IMAGE_SIDE):
            raise ValueError('左右の画像は512角RGBである必要があります')
        output = io.BytesIO()
        image.resize((FACE_SIDE, FACE_SIDE), Image.Resampling.BILINEAR).save(output, format='PNG')
        return output.getvalue()


def predict_rgb_reference(encoder, regressor, images):
    """左右の開始RGBを同じ縮小・特徴順で読み、scene座標mを返す。"""
    if len(images) != 2:
        raise ValueError('左右2枚の画像が必要です')
    tensors = []
    for image in images:
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format='PNG')
        tensors.append(face_image_tensor(face_rgb(stream.getvalue())))
    features = encode(encoder, torch.stack(tensors).contiguous())
    return regressor.predict(np.concatenate(features)[None])[0]/CANONICAL_MM_PER_M


class PandaRgb:
    def __init__(self, root, *, position_mode='legacy', locator_candidate=None, locator_input='shared-face', closed_position_head=None,all_position_head=None,diagnostic_ready_head=None):
        if all_position_head is not None and (closed_position_head is not None or locator_candidate is not None or position_mode!='full-stereo'):
            raise ValueError('全開閉の位置診断と別の位置出力を混在させません')
        if position_mode not in ('legacy', 'full-stereo', 'peak-stereo'):
            raise ValueError('画像位置の方式が未登録です')
        if locator_candidate is not None and position_mode != 'full-stereo':
            raise ValueError('画像位置の診断候補は一貫したステレオXYZでだけ評価します')
        if locator_input not in ('shared-face', 'resized'):
            raise ValueError('位置画像の前処理が未登録です')
        if locator_input == 'resized' and position_mode != 'full-stereo':
            raise ValueError('独立した位置画像は一貫したステレオXYZでだけ評価します')
        self.locator_input = locator_input
        self.position_mode = position_mode
        root = Path(root)
        calibration = verified_model_bytes(root/'c-open-v7/calibration.pkl', EXPECTED_MODELS['c-open-v7/calibration.pkl'])
        torch.set_num_threads(2)
        self.face = FaceFeatures(root/'separate-v2/openness', 'mps',
                                 expected_sha256=EXPECTED_MODELS['separate-v2/openness/weights.pt'])
        self.position = StereoPosition(root/'target-readout-v1', root/'rgb-depth-v1/position.pkl',
                                       baseline_sha256=EXPECTED_MODELS['rgb-depth-v1/position.pkl'],
                                       locator_sha256=EXPECTED_MODELS['target-readout-v1/weights.pt'])
        self.calibration = restore_sklearn_model(calibration)
        self.hashes = dict(EXPECTED_MODELS)
        self.ready_head=None
        if diagnostic_ready_head is not None:
            import copy
            folder=Path(diagnostic_ready_head);card=json.loads((folder/'result.json').read_bytes())
            if not card['completed'] or card['face_sha256']!=self.face.sha256:
                raise ValueError('開眼準備の診断候補が保存記録と異なります')
            sha=card['head_sha256'];content=verified_model_bytes(folder/'head.pt',sha)
            self.ready_head=copy.deepcopy(self.face.model.openness_head)
            self.ready_head.load_state_dict(torch.load(io.BytesIO(content),map_location=self.face.device,weights_only=True))
            self.ready_head.eval();self.hashes['diagnostic_ready/head.pt']=sha
        if locator_candidate is not None:
            self.position.locator = load_locator_candidate(locator_candidate)
            self.hashes['diagnostic_candidate/weights.pt'] = self.position.locator.sha256
        self.closed_position_model=None
        self.all_position_model=None
        if all_position_head is not None:
            folder=Path(all_position_head);card=json.loads((folder/'inference.json').read_bytes())
            if card.get('intended_position_source')!='canonical-all' or card.get('diagnostic_only') is not True:
                raise ValueError('全開閉の診断用位置モデルではありません')
            self.all_position_model=read_canonical_regressor(folder,self.face.sha256)
            self.hashes['diagnostic_all/model.pkl']=hashlib.sha256((folder/'diagnostic-model.pkl').read_bytes()).hexdigest()
        if closed_position_head is not None:
            if self.position_mode!='full-stereo':raise ValueError('閉眼位置候補は開眼時のステレオ出力を保持する構成でだけ使います')
            folder=Path(closed_position_head)
            self.closed_position_model=read_canonical_regressor(folder,self.face.sha256)
            self.hashes['diagnostic_closed/model.pkl']=hashlib.sha256((folder/'diagnostic-model.pkl').read_bytes()).hexdigest()

    def predict(self, left, right, *, native_face=None):
        if self.all_position_model is not None and native_face is None:
            raise ValueError('全開閉の位置診断は学習と同じnative256画像が必要です')
        small = face_rgb(left) if native_face is None else native_face
        if native_face is not None:
            face_image_tensor(small)
        face_rgb(right)
        features = self.face.extract(small).cpu().numpy()
        opening = float(self.calibration.predict([eye_image_features(small, features)])[0])
        if self.ready_head is not None:
            from .eye_ready_head import ready_from_image
            opening=ready_from_image(self.ready_head,small,features,self.face.device)
        if self.all_position_model is not None:
            point=np.asarray(self.all_position_model.predict(encode(self.face,face_image_tensor(small)[None]))[0])/CANONICAL_MM_PER_M
            if point.shape!=(3,) or not np.isfinite([*point,opening]).all():raise ValueError('全開閉の位置診断出力が契約外です')
            return {'eye_scene':point.tolist(),'full_stereo_eye_scene':None,'position_mode':self.position_mode,
                    'position_source':'canonical-all','locator_input':self.locator_input,'openness':opening,'stereo':None}
        if self.closed_position_model is not None and opening<CLOSED_POSITION_OPEN_THRESHOLD:
            point=np.asarray(self.closed_position_model.predict(encode(self.face,face_image_tensor(small)[None]))[0])/CANONICAL_MM_PER_M
            if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('閉眼位置推定が契約外です')
            return {'eye_scene':point.tolist(),'full_stereo_eye_scene':None,'position_mode':self.position_mode,
                    'position_source':'canonical-closed','locator_input':self.locator_input,'openness':opening,'stereo':None}
        position_image = face_rgb(left) if self.locator_input == 'resized' else small
        point, stereo = self.position.predict(features, position_image, left, right)
        if self.position_mode == 'full-stereo':
            point = np.array(stereo['measured_point'])
        elif self.position_mode == 'peak-stereo':
            with torch.inference_mode():
                latent = self.position.locator.model.image_features(face_image_tensor(small)[None].to('mps'))
                maps = self.position.locator.model.readout(latent)[:, :1]
                pixel = target_peak_coordinates(maps)[0].cpu().numpy()
            arrays = [np.array(Image.open(io.BytesIO(content)).convert('RGB'), dtype=float) for content in (left, right)]
            match = match_disparity(*arrays, pixel*IMAGE_SIDE-.5, image_side=IMAGE_SIDE, patch_radius=STEREO_PATCH_RADIUS)
            point = disparity_point(pixel*2-1, match['disparity'], image_side=IMAGE_SIDE)
            stereo = {'pixel': pixel.tolist(), 'measured_point': point.tolist(), **match}
        if not np.isfinite([*point, opening]).all():
            raise ValueError('画像推定に非有限値があります')
        return {'eye_scene': point.tolist(), 'full_stereo_eye_scene': stereo['measured_point'], 'position_mode': self.position_mode,
                'locator_input': self.locator_input,
                'openness': opening, 'stereo': stereo}
