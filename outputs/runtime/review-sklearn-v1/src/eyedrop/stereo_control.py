"""追加RGBから求めた奥行きだけを、固定した動作方策へ接続する。"""
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .face_features import FaceFeatures, face_image_tensor
from .target_supervision import target_coordinates
from .stereo_geometry import match_disparity, disparity_point
from .state_control import observation
from .model_integrity import FIXED_RGB_MODELS, verified_model_bytes, restore_sklearn_model

SIDE = 512
RADIUS = 8


class StereoPosition:
    def __init__(self, locator_folder, baseline_path, *,
                 baseline_sha256=FIXED_RGB_MODELS['rgb-depth-v1/position.pkl'],
                 locator_sha256=FIXED_RGB_MODELS['target-readout-v1/weights.pt']):
        folder = Path(locator_folder)
        report = json.loads((folder / 'stereo-report.json').read_bytes())
        if not report['eligible_for_geometry']:
            raise ValueError('2眼画像の診断に合格していません')
        if report['weights_sha256'] != locator_sha256:
            raise ValueError('固定した位置検出器のハッシュと一致しません')
        content = verified_model_bytes(baseline_path, baseline_sha256)
        self.locator = FaceFeatures(folder, 'mps', expected_sha256=locator_sha256)
        self.baseline = restore_sklearn_model(content)
        self.baseline_sha256 = baseline_sha256

    def predict(self, features, left, high_left, high_right):
        features = np.asarray(features, dtype=float)
        if features.shape != (9,) or not np.isfinite(features).all():
            raise ValueError('元の画像認識値が不正です')
        arrays = []
        for content in (high_left, high_right):
            with Image.open(io.BytesIO(content)) as image:
                if image.size != (SIDE, SIDE):
                    raise ValueError('2眼画像は512角が必要です')
                if image.mode == 'RGBA' and image.getchannel('A').getextrema() == (255, 255):
                    image = image.convert('RGB')
                if image.mode != 'RGB':
                    raise ValueError('2眼画像は不透明なRGBが必要です')
                arrays.append(np.asarray(image, dtype=float))
        with torch.inference_mode():
            latent = self.locator.model.image_features(face_image_tensor(left)[None].to('mps'))
            pixel = target_coordinates(self.locator.model.readout(latent)[:, :1])[0].cpu().numpy()
        match = match_disparity(*arrays, pixel * SIDE - .5, image_side=SIDE, patch_radius=RADIUS)
        measured = disparity_point(pixel * 2 - 1, match['disparity'], image_side=SIDE)
        point = self.baseline.baseline.predict(features[None])[0].copy()
        point[2] = measured[2]
        return point, {'pixel': pixel.tolist(), 'measured_point': measured.tolist(), **match}


class StereoController:
    def __init__(self, position, motion):
        self.position = position
        self.motion = motion

    def action(self, features, own, memory, left, high_left, high_right, calibrated_openness):
        point, diagnostic = self.position.predict(features, left, high_left, high_right)
        row = observation(point, own, float(calibrated_openness), memory)
        return {'action': self.motion.action(row), 'predicted_point': point.tolist(),
                'predicted_openness': float(calibrated_openness), 'observation': row.tolist(), 'stereo': diagnostic}
