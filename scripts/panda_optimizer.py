"""公式設定を選択し、再開時の最適化条件を照合する。"""
import torch

LEGACY = 'legacy-adamw'
OFFICIAL = 'official-smolvla'


def build_optimizer(config, parameters, mode, horizon, previous=None):
    if previous is not None:
        if previous.get('optimizer_mode', LEGACY) != mode:
            raise ValueError('再開時に最適化方式を変更できません')
        if mode == OFFICIAL and previous.get('scheduler_horizon') != horizon:
            raise ValueError('再開時に学習率の予定更新数を変更できません')
    if mode == LEGACY:
        return torch.optim.AdamW(parameters, lr=config.optimizer_lr, fused=False), None
    if mode != OFFICIAL or horizon <= 0:
        raise ValueError('最適化条件が不正です')
    optimizer = config.get_optimizer_preset().build(parameters)
    return optimizer, config.get_scheduler_preset().build(optimizer, horizon)
