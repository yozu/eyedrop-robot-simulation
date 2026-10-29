"""左右・高さを固定したまま、画像由来の奥行き回帰だけを交換する。"""
import numpy as np

DEPTH_AXIS = 2
LAYOUT_FEATURES = 8


class DepthPosition:
    def __init__(self, baseline, depth):
        self.baseline = baseline
        self.depth = depth

    def predict(self, features):
        features = np.asarray(features)
        points = self.baseline.predict(features).copy()
        points[:, DEPTH_AXIS] = self.depth.predict(features[:, :LAYOUT_FEATURES])
        return points
