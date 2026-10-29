"""7関節の情報を残した直交座標変換。次元削減や指令の補正は行わない。"""
import numpy as np

JOINT_DIM = 7
ABSOLUTE = 'absolute-joints-v1'
ORTHOGONAL = 'full-rank-joint-basis-v1'
MIN_AXIS_STD_RAD = 1e-8
ORTHOGONAL_TOLERANCE = 1e-8


def checked_basis(basis):
    mean = np.asarray(basis['mean'], dtype=np.float64)
    components = np.asarray(basis['components'], dtype=np.float64)
    if (mean.shape != (JOINT_DIM,) or components.shape != (JOINT_DIM, JOINT_DIM)
        or not np.isfinite(mean).all() or not np.isfinite(components).all()
        or not np.allclose(components@components.T, np.eye(JOINT_DIM), atol=ORTHOGONAL_TOLERANCE, rtol=0)):
        raise ValueError('7次元の有限な直交基底が必要です')
    return mean, components


def checked_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != JOINT_DIM or not np.isfinite(values).all():
        raise ValueError('有限の7関節が必要です')
    return values


def fit_basis(values):
    values = checked_values(values)
    if values.ndim != 2 or len(values) <= JOINT_DIM:
        raise ValueError('7次元より多い学習標本が必要です')
    mean = values.mean(0)
    _, singular, components = np.linalg.svd(values-mean, full_matrices=False)
    if np.any(singular/len(values)**.5 <= MIN_AXIS_STD_RAD):
        raise ValueError('学習関節の分散が不足しています')
    return {'mean': mean.tolist(), 'components': components.tolist(), 'source_split': 'train', 'dimensions': JOINT_DIM}


def encode_joints(values, basis):
    mean, components = checked_basis(basis)
    return (checked_values(values)-mean)@components.T


def decode_joints(values, basis):
    mean, components = checked_basis(basis)
    return checked_values(values)@components+mean
