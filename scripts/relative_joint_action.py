"""観測角基準の7関節と絶対指幅を、公式LeRobotの変換で接続する。"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from lerobot.processor.relative_action_processor import to_relative_actions, to_absolute_actions
from action_encoding import LINEAR
from cartesian_action import JOINT_SPACE
from joint_action_basis import ABSOLUTE

MODE = 'joint-relative-observation-v1'
ACTION_DIM = 8
JOINT_DIM = 7
STATE_DIM = 32
MASK = (True,) * JOINT_DIM + (False,)
ANCHOR = 'raw_observation.state[:7]_at_image_capture'
BINDING_FILE = 'eyedrop-action-contract.json'
UNITS = ['rad'] * JOINT_DIM + ['m_total_gap']
ORIGINAL_SCALE = 'fixed_original_train_action_std'
RELATIVE_SCALE = 'observed_relative_train_std'
SHA256_LENGTH = 64


def document_hash(document):
    payload = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build_normalization(source, source_sha256):
    if (source.get('source_split') != 'train' or
        source.get('joint_action_encoding', ABSOLUTE) != ABSOLUTE or
        source.get('action_encoding', LINEAR) != LINEAR or
        source.get('robot_action_space', JOINT_SPACE) != JOINT_SPACE):
        raise ValueError('元のtrain由来・絶対関節・線形指幅の統計が必要です')
    document = copy.deepcopy(source)
    document['joint_action_encoding'] = MODE
    document['action_encoding'] = LINEAR
    document['robot_action_space'] = JOINT_SPACE
    document['features']['action']['mean'][:JOINT_DIM] = [0.] * JOINT_DIM
    document['relative_action_contract'] = dict(
        version=1, anchor=ANCHOR, units=UNITS, mask=list(MASK),
        state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_size=1,
        scale_source=ORIGINAL_SCALE, source_sha256=source_sha256)
    validate_normalization(document)
    return document


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == SHA256_LENGTH and
            all(character in '0123456789abcdef' for character in value))


def _valid_scale_audit(audit):
    if not isinstance(audit, dict) or not _valid_hash(audit.get('sha256')):
        return False
    sources = audit.get('input_hashes')
    return (isinstance(sources, dict) and bool(sources) and
            all(isinstance(path, str) and '/train/data/' in path and _valid_hash(digest)
                for path, digest in sources.items()))


def apply_relative_train_scale(document, statistics, statistics_sha256):
    validate_normalization(document)
    if (document['relative_action_contract']['scale_source'] != ORIGINAL_SCALE or
        statistics.get('completed') is not True or statistics.get('updates') != 0 or
        statistics.get('scale_source_train_only') is not True or
        statistics.get('current_joint_scale') != document['features']['action']['std'][:JOINT_DIM]):
        raise ValueError('元の尺度と対応する学習データの監査が必要です')
    audit = dict(sha256=statistics_sha256, input_hashes=copy.deepcopy(statistics.get('input_hashes')))
    scales = statistics.get('relative_train_std')
    if not _valid_scale_audit(audit) or not isinstance(scales, list) or len(scales) != JOINT_DIM:
        raise ValueError('7関節の尺度と学習元のハッシュが必要です')
    result = copy.deepcopy(document)
    result['features']['action']['std'][:JOINT_DIM] = scales
    result['relative_action_contract'].update(scale_source=RELATIVE_SCALE, scale_audit=audit)
    validate_normalization(result)
    return result


def validate_normalization(document):
    contract = document.get('relative_action_contract', {})
    stats = document['features']['action']
    mean, scale = torch.as_tensor(stats['mean']), torch.as_tensor(stats['std'])
    scale_source = contract.get('scale_source')
    if ((scale_source == ORIGINAL_SCALE and 'scale_audit' in contract) or
        (scale_source == RELATIVE_SCALE and not _valid_scale_audit(contract.get('scale_audit')))):
        raise ValueError('相対量の尺度の監査元が不明または矛盾しています')
    if (document.get('joint_action_encoding') != MODE or document.get('source_split') != 'train' or
        document.get('action_encoding', LINEAR) != LINEAR or
        document.get('robot_action_space', JOINT_SPACE) != JOINT_SPACE or
        document.get('joint_action_basis') is not None or
        contract.get('anchor') != ANCHOR or contract.get('version') != 1 or
        contract.get('units') != UNITS or contract.get('mask') != list(MASK) or
        contract.get('state_dim') != STATE_DIM or contract.get('action_dim') != ACTION_DIM or
        contract.get('chunk_size') != 1 or scale_source not in (ORIGINAL_SCALE, RELATIVE_SCALE) or
        len(contract.get('source_sha256', '')) != 64 or
        mean.shape != (ACTION_DIM,) or scale.shape != (ACTION_DIM,) or
        not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or
        not (scale > 0).all() or torch.count_nonzero(mean[:JOINT_DIM])):
        raise ValueError('相対関節の尺度・単位・基準時点が契約と異なります')


def _convert(action, state, function):
    if state is None:
        raise ValueError('対応する観測時点の生の自己状態が必要です')
    action = torch.as_tensor(action, dtype=torch.float32)
    state = torch.as_tensor(state, dtype=torch.float32, device=action.device)
    if (action.shape not in ((ACTION_DIM,), (1, ACTION_DIM)) or state.shape != (STATE_DIM,) or
        not torch.isfinite(action).all() or not torch.isfinite(state).all()):
        raise ValueError('相対指令は単一観測32成分と単一動作8成分に限定します')
    return function(action.reshape(1, ACTION_DIM), state.unsqueeze(0), MASK).reshape(action.shape)


def relative_action(action, state):
    return _convert(action, state, to_relative_actions)


def absolute_action(action, state):
    return _convert(action, state, to_absolute_actions)


def bind_policy(policy, document):
    validate_normalization(document)
    if policy.config.chunk_size != 1 or policy.config.n_action_steps != 1:
        raise ValueError('相対指令は1予測1実行のモデルだけです')
    policy._eyedrop_action_contract = dict(mode=MODE, normalization_sha256=document_hash(document))


def verify_initial_policy(directory, document, expected_sha256):
    """相対指令へ対応付ける学習前の重みを、モデル読込前に照合する。"""
    validate_normalization(document)
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_bytes())
    if ((directory / BINDING_FILE).exists() or
        config.get('chunk_size') != 1 or
        config.get('output_features', {}).get('action', {}).get('shape') != [ACTION_DIM] or
        file_hash(directory / 'model.safetensors') != expected_sha256):
        raise ValueError('相対指令へ対応付ける未変更の初期重み・動作数が一致しません')


def validate_policy_binding(policy, document):
    binding = getattr(policy, '_eyedrop_action_contract', None)
    if document.get('joint_action_encoding') == MODE:
        validate_normalization(document)
        if (not binding or binding != dict(mode=MODE, normalization_sha256=document_hash(document)) or
            policy.config.chunk_size != 1 or policy.config.n_action_steps != 1):
            raise ValueError('相対指令モデルと正規化・実行数の対応が不一致です')
    elif binding is not None:
        raise ValueError('相対指令モデルを絶対指令として復元できません')


def save_policy_binding(policy, directory):
    binding = getattr(policy, '_eyedrop_action_contract', None)
    if not binding:
        raise ValueError('相対指令の保存契約がありません')
    directory = Path(directory)
    document = dict(**binding, model_sha256=file_hash(directory / 'model.safetensors'))
    (directory / BINDING_FILE).write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


def read_policy_binding(directory):
    path = Path(directory) / BINDING_FILE
    if path.exists():
        document = json.loads(path.read_bytes())
        if (set(document) != {'mode', 'normalization_sha256', 'model_sha256'} or
            document['mode'] != MODE or
            document['model_sha256'] != file_hash(Path(directory) / 'model.safetensors')):
            raise ValueError('保存した相対指令の重み・設定が一致しません')
        return document
    return None


def load_policy_binding(policy, directory):
    document = read_policy_binding(directory)
    if document is not None:
        policy._eyedrop_action_contract = {k: document[k] for k in ('mode', 'normalization_sha256')}
