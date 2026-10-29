"""開始RGBと現在RGBから、位置保持に必要な観測の有効性だけを判定する。"""
import cv2
import math
import numpy as np
from eyedrop.camera_position import CAMERA, FOV_DEGREES, basis
from eyedrop.stereo_geometry import BASELINE

SIDE = 512
OPEN_THRESHOLD = .95
ROI_RADIUS_M = .06
EYE_STRIP_RADIUS_M = .012
MOTION_BUDGET_M = .00025
MIN_POINTS = 8
MAX_POINTS = 80
FEATURE_QUALITY = .01
FEATURE_DISTANCE_PX = 5
FORWARD_BACKWARD_PX = .5
LK_WINDOW = (21, 21)
LK_LEVELS = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .001)
DENSE_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
DENSE_MIN_CORRELATION = .99
DENSE_MIN_PIXELS = 64
DENSE_GAUSSIAN_SIZE = 5


def pixels_and_jacobian(point):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('RGB由来の開始位置が不正です')
    forward, right, up = basis(); focal = SIDE / 2 / np.tan(np.deg2rad(FOV_DEGREES / 2))
    pixels = []; derivatives = []
    for origin in (CAMERA, CAMERA + right * BASELINE):
        delta = point-origin; depth = delta @ forward
        if depth <= 0:
            raise ValueError('開始位置がカメラ後方です')
        x, y = delta @ right, delta @ up
        pixels.append([SIDE/2+focal*x/depth, SIDE/2-focal*y/depth])
        derivatives.extend([focal/depth*(right-x/depth*forward), focal/depth*(-up+y/depth*forward)])
    return np.array(pixels), np.array(derivatives)


def gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape not in ((SIDE, SIDE), (SIDE, SIDE, 3)):
        raise ValueError('観測画像は512角のuint8です')
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


class ImageMotionValidity:
    def __init__(self, images, masks, point):
        if len(images) != 2 or len(masks) != 2:
            raise ValueError('左右の観測が必要です')
        centers, jacobian = pixels_and_jacobian(point)
        self.threshold_px = float(np.linalg.svd(jacobian, compute_uv=False).min() * MOTION_BUDGET_M)
        self.references = [gray(x) for x in images]; self.points = []
        forward, _, _ = basis(); depth = (np.asarray(point)-CAMERA) @ forward
        focal = SIDE / 2 / np.tan(np.deg2rad(FOV_DEGREES / 2))
        radius = focal * ROI_RADIUS_M / depth; strip = focal * EYE_STRIP_RADIUS_M / depth
        yy, xx = np.mgrid[:SIDE, :SIDE]
        for image, mask, (x, y) in zip(self.references, masks, centers):
            mask = self._mask(mask)
            allowed = ((xx-x)**2+(yy-y)**2 <= radius**2) & (np.abs(yy-y) > strip) & ~mask
            self.points.append(cv2.goodFeaturesToTrack(image, maxCorners=MAX_POINTS, qualityLevel=FEATURE_QUALITY,
                minDistance=FEATURE_DISTANCE_PX, mask=allowed.astype(np.uint8)*255))

    @staticmethod
    def _mask(mask):
        mask = np.asarray(mask)
        if mask.shape != (SIDE, SIDE) or mask.dtype != bool:
            raise ValueError('自己マスクの形が不正です')
        return mask

    def check(self, images, masks):
        if len(images) != 2 or len(masks) != 2:
            raise ValueError('左右の観測が必要です')
        shifts = []; counts = []
        for reference, points, image, mask in zip(self.references, self.points, images, masks):
            image = gray(image); mask = self._mask(mask)
            if points is None or len(points) < MIN_POINTS:
                return {'valid': False, 'reason': 'initial_features_missing', 'threshold_px': self.threshold_px}
            moved, status, _ = cv2.calcOpticalFlowPyrLK(reference, image, points, None, winSize=LK_WINDOW,
                maxLevel=LK_LEVELS, criteria=LK_CRITERIA)
            if moved is None:
                return {'valid': False, 'reason': 'tracking_failed', 'threshold_px': self.threshold_px}
            back, backward_status, _ = cv2.calcOpticalFlowPyrLK(image, reference, moved, None, winSize=LK_WINDOW,
                maxLevel=LK_LEVELS, criteria=LK_CRITERIA)
            if back is None:
                return {'valid': False, 'reason': 'backward_tracking_failed', 'threshold_px': self.threshold_px}
            coordinates = moved[:, 0]; integer = np.rint(coordinates).astype(int)
            inside = np.all((integer >= 0) & (integer < SIDE), axis=1)
            visible = np.zeros(len(points), dtype=bool)
            visible[inside] = ~mask[integer[inside, 1], integer[inside, 0]]
            valid = (status[:, 0] == 1) & (backward_status[:, 0] == 1) & visible & (np.linalg.norm(back-points, axis=2)[:, 0] <= FORWARD_BACKWARD_PX)
            counts.append(int(valid.sum()))
            if valid.sum() < MIN_POINTS:
                return {'valid': False, 'reason': 'current_observation_missing', 'counts': counts, 'threshold_px': self.threshold_px}
            shifts.extend(np.median((moved-points)[valid, 0], axis=0).tolist())
        motion = float(np.linalg.norm(shifts))
        return {'valid': motion <= self.threshold_px, 'reason': 'within_threshold' if motion <= self.threshold_px else 'image_motion',
                'motion_px': motion, 'shifts_px': shifts, 'threshold_px': self.threshold_px, 'counts': counts}


def dense_check(references, reference_masks, images, masks, point):
    if any(len(items) != 2 for items in (references, reference_masks, images, masks)):
        raise ValueError('左右の観測が必要です')
    centers, jacobian = pixels_and_jacobian(point)
    threshold = float(np.linalg.svd(jacobian, compute_uv=False).min()*MOTION_BUDGET_M)
    forward, _, _ = basis(); depth = (np.asarray(point)-CAMERA) @ forward
    focal = SIDE/2/np.tan(np.deg2rad(FOV_DEGREES/2))
    radius = focal*ROI_RADIUS_M/depth; strip = focal*EYE_STRIP_RADIUS_M/depth
    yy, xx = np.mgrid[:SIDE, :SIDE]; shifts = []; correlations = []; counts = []
    for reference, old_mask, image, mask, (x, y) in zip(references, reference_masks, images, masks, centers):
        reference = gray(reference); image = gray(image)
        allowed = ((xx-x)**2+(yy-y)**2 <= radius**2) & (np.abs(yy-y) > strip)
        allowed &= ~ImageMotionValidity._mask(old_mask) & ~ImageMotionValidity._mask(mask)
        counts.append(int(allowed.sum()))
        if allowed.sum() < DENSE_MIN_PIXELS or np.var(reference[allowed]) == 0 or np.var(image[allowed]) == 0:
            return {'valid': False, 'reason': 'dense_observation_missing', 'counts': counts, 'threshold_px': threshold}
        rows, columns = np.where(allowed)
        y0, y1 = int(rows.min()), int(rows.max())+1; x0, x1 = int(columns.min()), int(columns.max())+1
        selection = np.s_[y0:y1, x0:x1]
        try:
            correlation, transform = cv2.findTransformECC(reference[selection], image[selection], np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_TRANSLATION, DENSE_CRITERIA, allowed[selection].astype(np.uint8)*255, DENSE_GAUSSIAN_SIZE)
        except cv2.error as error:
            if error.code != cv2.Error.StsNoConv:
                raise
            return {'valid': False, 'reason': 'dense_nonconvergence', 'counts': counts, 'threshold_px': threshold}
        if not np.isfinite(transform).all() or not np.isfinite(correlation):
            raise ValueError('領域照合の出力が非有限です')
        correlations.append(float(correlation)); shifts.extend(transform[:, 2].tolist())
        if correlation < DENSE_MIN_CORRELATION:
            return {'valid': False, 'reason': 'dense_correlation_low', 'correlations': correlations, 'counts': counts, 'threshold_px': threshold}
    motion = float(np.linalg.norm(shifts))
    return {'valid': motion <= threshold, 'reason': 'within_threshold' if motion <= threshold else 'image_motion',
        'motion_px': motion, 'shifts_px': shifts, 'correlations': correlations, 'counts': counts, 'threshold_px': threshold}


class ObservedReferencePosition:
    def __init__(self, predict_point, *, max_age_seconds):
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError('診断の参照時間上限が不正です')
        self.predict_point = predict_point
        self.max_age_seconds = max_age_seconds
        self.reference = None
        self.previous_time = None

    def update(self, images, masks, opening, time_s):
        if not math.isfinite(opening) or not 0 <= opening <= 1 or not math.isfinite(time_s) or time_s < 0 or self.previous_time is not None and time_s < self.previous_time:
            raise ValueError('観測の開眼度または時刻が不正です')
        self.previous_time = time_s
        if opening < OPEN_THRESHOLD:
            return {'valid': False, 'point_scene': None, 'reason': 'eye_not_open'}
        if self.reference is None:
            point = np.asarray(self.predict_point(images), dtype=float)
            pixels_and_jacobian(point)
            self.reference = ([np.array(x).copy() for x in images], [np.array(m).copy() for m in masks], point.copy(), time_s)
        initial, initial_masks, point, started = self.reference
        if time_s-started > self.max_age_seconds:
            return {'valid': False, 'point_scene': None, 'reason': 'diagnostic_reference_expired'}
        registration = dense_check(initial, initial_masks, images, masks, point)
        return {'valid': registration['valid'], 'point_scene': point.tolist() if registration['valid'] else None,
                'reason': registration['reason'], 'registration': registration, 'reference_age_s': time_s-started}
