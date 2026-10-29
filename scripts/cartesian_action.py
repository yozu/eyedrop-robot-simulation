"""ノズルXYZと指幅の明示した4次元契約。ロボットFK/IKや眼の真値は含めない。"""
import copy
import numpy as np
from action_encoding import encode_gap, decode_gap, BOUNDED

CARTESIAN_SPACE = 'nozzle-xyz-gap-v1'
JOINT_SPACE = 'panda-joints-gap-v1'
CARTESIAN_DIM = 4
JOINT_DIM = 8
STATE_DIM = 32
XYZ_STD_FLOOR_M = .001


def checked(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != CARTESIAN_DIM or not np.isfinite(values).all():
        raise ValueError('Cartesian動作の形または値が不正です')
    return values


def encode_cartesian(values):
    result = checked(values).copy()
    result[...,-1] = encode_gap(result[...,-1])
    return result


def decode_cartesian(values):
    result = checked(values).copy()
    result[...,-1] = decode_gap(result[...,-1])
    return result


def action_dimension(document):
    space = document.get('robot_action_space', JOINT_SPACE)
    if space == JOINT_SPACE: return JOINT_DIM
    if space != CARTESIAN_SPACE: raise ValueError('未登録の動作空間です')
    if document.get('joint_action_basis') is not None or document.get('joint_action_encoding') not in (None, 'absolute-joints-v1'):
        raise ValueError('Cartesian動作と関節基底を混ぜられません')
    if document.get('action_encoding') != BOUNDED or document.get('source_split') != 'train':
        raise ValueError('Cartesian動作の符号化・統計分割が不正です')
    stats = document['features']['action']
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    if mean.shape != (CARTESIAN_DIM,) or std.shape != mean.shape or not np.isfinite([mean,std]).all() or (std <= 0).any():
        raise ValueError('Cartesian動作の統計が不正です')
    return CARTESIAN_DIM


def normalization_from_training(actions, baseline):
    actions = checked(actions)
    if actions.ndim != 2 or len(actions) == 0: raise ValueError('学習時点の配列が必要です')
    encode_cartesian(actions)
    if baseline['source_split'] != 'train' or baseline['action_encoding'] != BOUNDED:
        raise ValueError('元統計の分割または指幅表現が不正です')
    state = baseline['features']['observation.state']
    if len(state['mean']) != STATE_DIM or len(state['std']) != STATE_DIM: raise ValueError('元の自己状態32成分が必要です')
    previous = baseline['features']['action']
    result = {'source_split': 'train', 'robot_action_space': CARTESIAN_SPACE, 'action_encoding': BOUNDED,
              'xyz_std_floor_m': XYZ_STD_FLOOR_M,
              'features': {'observation.state': copy.deepcopy(state), 'action': {
                  'mean': [*actions[:,:3].mean(axis=0).tolist(), previous['mean'][-1]],
                  'std': [*np.maximum(actions[:,:3].std(axis=0), XYZ_STD_FLOOR_M).tolist(), previous['std'][-1]]}}}
    action_dimension(result)
    return result


def check_action_resume(previous, document):
    dimension = action_dimension(document)
    if (previous.get('robot_action_space', JOINT_SPACE) != document.get('robot_action_space', JOINT_SPACE)
            or previous.get('action_dim', JOINT_DIM) != dimension):
        raise ValueError('再開時に動作空間または次元を変更できません')
    return True
