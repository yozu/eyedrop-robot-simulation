"""SmolVLAの推論を変えず、切替対の差または公式の多時刻損失を学習する。"""
from contextlib import contextmanager
import math

import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from paired_switch_data import PAIR_DIFFERENCE, PAIR_FLOW_GRID, PAIR_OBJECTIVES, FLOW_TIMES, FLOW_STEPS

ACTION_DIM = 8
PADDED_DIM = 32
STATE_DIM = 32
PAIR_SIZE = 2
CHUNK_SIZE = 1
GAP_INDEX = ACTION_DIM - 1
ENDPOINT_TIME = 1.
OBSERVATION_KEYS = {'observation.state', 'observation.images.camera1', 'observation.images.camera2',
                    OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK}
TRAIN_KEYS = OBSERVATION_KEYS | {'action', 'action_is_pad'}


def pair_gap_loss(estimated, targets):
    shape = (PAIR_SIZE, CHUNK_SIZE, ACTION_DIM)
    if (estimated.shape != shape or targets.shape != shape
            or not torch.isfinite(estimated).all() or not torch.isfinite(targets).all()):
        raise ValueError('有限な2時点・1動作・8成分が必要です')
    difference = estimated[1, 0, GAP_INDEX] - estimated[0, 0, GAP_INDEX]
    target = targets[1, 0, GAP_INDEX].detach() - targets[0, 0, GAP_INDEX].detach()
    return (difference - target).square() / ACTION_DIM


def _validate(policy, batch, generator):
    cfg = policy.config
    if (tuple(cfg.action_feature.shape) != (ACTION_DIM,) or cfg.max_action_dim != PADDED_DIM
            or cfg.chunk_size != CHUNK_SIZE or set(batch) != TRAIN_KEYS):
        raise ValueError('単時点8/32成分と許可した観測・教師だけを扱います')
    if batch['action'].shape != (PAIR_SIZE, CHUNK_SIZE, ACTION_DIM):
        raise ValueError('対の教師指令が不正です')
    if batch['observation.state'].shape != (PAIR_SIZE, STATE_DIM):
        raise ValueError('対の自己状態は32成分です')
    pad = batch['action_is_pad']
    if pad.shape != (PAIR_SIZE, CHUNK_SIZE) or pad.dtype != torch.bool or pad.any():
        raise ValueError('実在する隣接2時点だけを使います')
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != PAIR_SIZE or not torch.isfinite(value).all():
            raise ValueError('対の観測・教師が不正です: ' + key)
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('対の抽選専用CPU生成器が必要です')


def paired_flow_loss(policy, batch, generator, *, objective=PAIR_DIFFERENCE, flow_time=None, official_forward=None):
    _validate(policy, batch, generator)
    if objective not in PAIR_OBJECTIVES:
        raise ValueError('登録されていない対の学習方式です')
    if objective == PAIR_FLOW_GRID:
        if flow_time not in FLOW_TIMES or policy.config.num_steps != FLOW_STEPS:
            raise ValueError('固定した10段推論の時刻だけを使います')
    elif flow_time not in (None, ENDPOINT_TIME):
        raise ValueError('旧方式はt=1だけを使います')
    observation = {key:batch[key] for key in OBSERVATION_KEYS}
    target = batch['action'].detach()
    device = target.device
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if device.type == 'mps' else None
    captured = []
    handle = None
    try:
        noise = torch.randn((1, CHUNK_SIZE, PADDED_DIM), generator=generator,
                            dtype=target.dtype).repeat(PAIR_SIZE, 1, 1).to(device)
        if objective == PAIR_FLOW_GRID:
            # 学習用の教師補間と有効8成分の損失は公式関数へ委ねる。
            times = torch.full((PAIR_SIZE,), flow_time, device=device, dtype=target.dtype)
            forward = official_forward if official_forward is not None else policy.forward
            loss, _ = forward(batch, noise=noise, time=times)
            return loss
        images, masks = policy.prepare_images(observation)
        state = policy.prepare_state(observation)
        # 教師動作をforwardへ渡さず、t=1で同じ乱数から始める。
        actions = policy.prepare_action({'action':torch.zeros_like(target)})
        times = torch.full((PAIR_SIZE,), ENDPOINT_TIME, device=device, dtype=target.dtype)
        handle = policy.model.action_out_proj.register_forward_hook(
            lambda module, arguments, output:captured.append(output))
        policy.model.forward(images, masks, observation[OBS_LANGUAGE_TOKENS],
                             observation[OBS_LANGUAGE_ATTENTION_MASK], state, actions, noise, times)
        if len(captured) != 1 or captured[0].shape != (PAIR_SIZE, CHUNK_SIZE, PADDED_DIM):
            raise ValueError('公式の動作出力を1回だけ取得できません')
        estimate = (noise - captured[0])[..., :ACTION_DIM]
        return pair_gap_loss(estimate, target)
    finally:
        if handle is not None:
            handle.remove()
        torch.set_rng_state(cpu_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


@contextmanager
def use_pair_loss(policy, pair_batch, generator, *, weight=1., objective=PAIR_DIFFERENCE, flow_time=None):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('追加係数は有限かつ非負です')
    if hasattr(policy, '_eyedrop_pair_loss_active'):
        raise ValueError('対の損失を二重に接続できません')
    stats = {'pair_calls':0}
    if weight == 0:
        yield stats
        return
    _validate(policy, pair_batch, generator)
    original = policy.forward

    def forward(batch, *args, **kwargs):
        if not policy.training or args or kwargs.get('reduction', 'mean') != 'mean':
            raise ValueError('通常の学習用平均損失だけに追加できます')
        official, logs = original(batch, **kwargs)
        extra = paired_flow_loss(policy, pair_batch, generator, objective=objective,
                                 flow_time=flow_time, official_forward=original)
        total = official + weight * extra
        if not torch.isfinite(total):
            raise ValueError('損失が非有限です')
        stats['pair_calls'] += 1
        stats.update(last_official_loss=float(official.detach()), last_pair_loss=float(extra.detach()))
        label = 'pair_flow_loss' if objective == PAIR_FLOW_GRID else 'pair_gap_loss'
        return total, dict(logs, official_loss=float(official.detach()),
                           **{label:float(extra.detach())}, loss=float(total.detach()))

    policy._eyedrop_pair_loss_active = True
    policy.forward = forward
    try:
        yield stats
    finally:
        policy.forward = original
        del policy._eyedrop_pair_loss_active
