"""予測した眼の画素と奥行きを、固定カメラの幾何で結ぶ。"""
import numpy as np

CAMERA = np.array([.12, 1.55, .6])
LOOK_AT = np.array([0., 1.20, -.1])
FOV_DEGREES = 40.
WORLD_UP = np.array([0., 1., 0.])
DEPTH_AXIS = 2
PARALLEL_TOLERANCE = 1e-10


def basis():
    forward = LOOK_AT - CAMERA
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, WORLD_UP)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def project(point):
    forward, right, up = basis()
    delta = np.asarray(point, dtype=float) - CAMERA
    scale = (delta @ forward) * np.tan(np.deg2rad(FOV_DEGREES / 2))
    return np.array([delta @ right, -delta @ up]) / scale


def point_at_depth(pixel, depth):
    pixel = np.asarray(pixel, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(depth):
        raise ValueError('有限の眼の画像座標2成分と奥行きが必要です')
    forward, right, up = basis()
    direction = forward + np.tan(np.deg2rad(FOV_DEGREES / 2)) * (pixel[0] * right - pixel[1] * up)
    if abs(direction[DEPTH_AXIS]) <= PARALLEL_TOLERANCE:
        raise ValueError('視線が奥行き平面と平行です')
    distance = (depth - CAMERA[DEPTH_AXIS]) / direction[DEPTH_AXIS]
    if distance <= 0:
        raise ValueError('カメラ後方の位置は使えません')
    return CAMERA + distance * direction


class CameraPosition:
    def __init__(self, depth_model):
        self.depth_model = depth_model

    def predict_image(self, features, content):
        depth = self.depth_model.predict_image(features, content)[DEPTH_AXIS]
        return point_at_depth(np.asarray(features)[:2], depth)
