"""画像から推定した顔特徴を、固定した状態方策へ接続する。"""
import numpy as np
from .state_control import observation


class PerceptionControl:
    def __init__(self, position_model, motion_model):
        self.position_model = position_model
        self.motion_model = motion_model

    def action(self, predicted_face, own, memory, image=None, *, calibrated_openness=None):
        features = np.asarray(predicted_face, dtype=float)
        if features.shape != (9,) or not np.isfinite(features).all():
            raise ValueError('予測顔特徴は有限の9成分が必要です')
        if image is None:
            point = self.position_model.predict(features.reshape(1, -1))[0]
        else:
            point = self.position_model.predict_image(features, image)
        openness = (features[8] + 1.) / 2.
        if calibrated_openness is not None:
            openness = float(calibrated_openness)
        row = observation(point, own, openness, memory)
        return {'action':self.motion_model.action(row), 'predicted_point':point.tolist(),
                'predicted_openness':float(openness), 'observation':row.tolist()}
