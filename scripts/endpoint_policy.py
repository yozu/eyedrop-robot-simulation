"""SmolVLA動作expertを、同じ入出力の決定論的1段回帰へ適応する。"""
import json
from pathlib import Path
from resumable_job import atomic_json

MODE = 'smolvla-endpoint-bc-v1'
REAL_DIM = 4
PADDED_DIM = 32
ENDPOINT_STEPS = 1
ORIGINAL_STEPS = 10
METADATA_NAME = 'task-policy.json'
METADATA = {'mode': MODE, 'num_steps': ENDPOINT_STEPS, 'noise': 'zero',
            'chunk_size': 1, 'action_dim': REAL_DIM, 'padded_dim': PADDED_DIM}


def configure_endpoint(policy):
    if (policy.config.chunk_size != 1 or policy.config.max_action_dim != PADDED_DIM
        or policy.config.action_feature.shape[0] != REAL_DIM
        or policy.config.num_steps not in (ORIGINAL_STEPS, ENDPOINT_STEPS)):
        raise ValueError('1動作・実4成分・32paddingの既知SmolVLAだけが対象です')
    policy.config.num_steps = ENDPOINT_STEPS
    policy.model.config.num_steps = ENDPOINT_STEPS


def endpoint_loss(policy, batch, gap_weight):
    import torch
    from smolvla_training import sequence_loss
    if policy.config.num_steps != ENDPOINT_STEPS:
        raise ValueError('先に1段回帰の推論条件を設定してください')
    state = batch['observation.state']
    noise = torch.zeros(state.shape[0], 1, PADDED_DIM, device=state.device, dtype=state.dtype)
    time = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
    return sequence_loss(policy, batch, 1., noise=noise, time=time, gap_weight=gap_weight)


def write_endpoint_metadata(directory):
    atomic_json(Path(directory)/METADATA_NAME, METADATA)


def validate_endpoint_policy(policy, directory, *, requested):
    metadata = Path(directory)/METADATA_NAME if directory is not None else None
    if requested:
        if metadata is not None:
            if not metadata.is_file() or json.loads(metadata.read_text()) != METADATA:
                raise ValueError('保存済み1段回帰の推論契約が一致しません')
            if policy.config.num_steps != ENDPOINT_STEPS:
                raise ValueError('保存モデルの積分数が1ではありません')
        configure_endpoint(policy)
    elif (metadata is not None and metadata.exists()) or policy.config.num_steps == ENDPOINT_STEPS:
        raise ValueError('1段回帰の重みを通常の乱数生成で実行できません')
