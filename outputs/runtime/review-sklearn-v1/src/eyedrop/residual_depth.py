"""画像から推定した顔配置と奥行きによる残差補正。"""

import numpy as np


class ResidualDepthPosition:
    def __init__(self, baseline, correction):
        self.baseline = baseline
        self.correction = correction

    def predict_image(self, features, content):
        point = self.baseline.predict_image(features, content).copy()
        face_coordinates = np.asarray(features)[:8]
        inputs = np.r_[face_coordinates, point[2]].reshape(1, -1)
        point[2] += self.correction.predict(inputs)[0]
        return point
