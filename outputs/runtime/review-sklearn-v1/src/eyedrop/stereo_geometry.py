"""平行な2台のRGBカメラで対応点と奥行きを計算する。"""
import numpy as np
from .camera_position import CAMERA, FOV_DEGREES, basis

SIDE = 256
BASELINE = .16
PATCH_RADIUS = 4
MIN_CAMERA_DEPTH = .35
MAX_CAMERA_DEPTH = 1.5
MIN_VARIANCE = 1e-12
FOCAL = (SIDE / 2) / np.tan(np.deg2rad(FOV_DEGREES / 2))


def sample(image, x, y):
    image = np.asarray(image, dtype=float)
    if np.any(x < 0) or np.any(y < 0) or np.any(x >= image.shape[1] - 1) or np.any(y >= image.shape[0] - 1):
        raise ValueError('対応点パッチが画像外です')
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - left, y - top
    if image.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    return ((1 - wx) * (1 - wy) * image[top, left] + wx * (1 - wy) * image[top, left + 1]
            + (1 - wx) * wy * image[top + 1, left] + wx * wy * image[top + 1, left + 1])


def match_disparity(left, right, pixel, disparities=None, *, image_side=SIDE, patch_radius=PATCH_RADIUS):
    pixel = np.asarray(pixel, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError('眼の画像座標が不正です')
    if image_side <= 0 or patch_radius < 1:
        raise ValueError('画像サイズと照合範囲が不正です')
    y, x = np.mgrid[-patch_radius:patch_radius + 1, -patch_radius:patch_radius + 1]
    x, y = x + pixel[0], y + pixel[1]
    target = sample(left, x, y).flatten()
    target = target - target.mean()
    norm = np.linalg.norm(target)
    if norm < MIN_VARIANCE:
        raise ValueError('眼パッチの模様が不足しています')
    if disparities is None:
        focal = FOCAL * image_side / SIDE
        disparities = range(int(np.ceil(focal * BASELINE / MAX_CAMERA_DEPTH)), int(np.floor(focal * BASELINE / MIN_CAMERA_DEPTH)) + 1)
    scores = []
    for disparity in disparities:
        if np.any(x - disparity < 0) or np.any(x - disparity >= np.asarray(right).shape[1] - 1):
            continue
        patch = sample(right, x - disparity, y).flatten()
        patch = patch - patch.mean()
        denominator = norm * np.linalg.norm(patch)
        if denominator < MIN_VARIANCE:
            continue
        scores.append((float(disparity), float(target @ patch / denominator)))
    if len(scores) < 3:
        raise ValueError('探索できる対応点が不足しています')
    best = int(np.argmax([s[1] for s in scores]))
    if best in (0, len(scores) - 1):
        raise ValueError('視差の最良点が探索端です')
    a, b, c = scores[best - 1:best + 2]
    if b[0] - a[0] != 1 or c[0] - b[0] != 1:
        raise ValueError('視差の補間に必要な連続点がありません')
    curvature = a[1] - 2 * b[1] + c[1]
    if curvature >= -MIN_VARIANCE:
        raise ValueError('視差のピークが不明瞭です')
    offset = .5 * (a[1] - c[1]) / curvature
    if abs(offset) > 1:
        raise ValueError('視差の補間が範囲外です')
    return {'disparity': b[0] + offset, 'ncc': b[1], 'integer_disparity': b[0]}


def disparity_point(normalized_pixel, disparity, *, image_side=SIDE):
    pixel = np.asarray(normalized_pixel, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(disparity) or disparity <= 0:
        raise ValueError('画像座標または視差が不正です')
    forward, right, up = basis()
    direction = forward + np.tan(np.deg2rad(FOV_DEGREES / 2)) * (pixel[0] * right - pixel[1] * up)
    return CAMERA + (FOCAL * image_side / SIDE * BASELINE / disparity) * direction
