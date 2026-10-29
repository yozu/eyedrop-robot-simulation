"""固定SmolVLAの画像・自己状態の特徴から、指幅を決定的に読み出す。"""
import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

HIDDEN_DIM = 64
STATE_DIM = 32
ALLOWED_KEYS = {'observation.state', 'observation.images.camera1', 'observation.images.camera2',
                OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK}


def observation_features(policy, batch):
    if set(batch) != ALLOWED_KEYS:
        raise ValueError('画像・自己状態・固定指示以外の入力は使えません')
    if policy.config.chunk_size != 1 or batch['observation.state'].shape != (1, STATE_DIM):
        raise ValueError('1観測・1動作・自己状態32成分が必要です')
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    device = batch['observation.state'].device
    dummy = torch.zeros(1, 1, policy.config.action_feature.shape[0], device=device)
    actions = policy.prepare_action({'action': dummy})
    captured = []
    handle = policy.model.action_out_proj.register_forward_pre_hook(
        lambda module, arguments: captured.append(arguments[0].detach().cpu()))
    try:
        with torch.no_grad():
            policy.model.forward(images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                                 state, actions, torch.zeros_like(actions), torch.ones(1, device=device))
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != (1, 1):
        raise ValueError('動作expertの特徴を一意に取得できません')
    features = torch.cat((batch['observation.state'].detach().cpu(), captured[0][:, 0]), dim=1).float()
    if not torch.isfinite(features).all():
        raise ValueError('特徴が非有限です')
    return features


class DirectGripperHead(torch.nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if (mean.ndim != 1 or mean.shape != scale.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('特徴の学習統計が不正です')
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.network = torch.nn.Sequential(torch.nn.Linear(len(mean), HIDDEN_DIM), torch.nn.SiLU(), torch.nn.Linear(HIDDEN_DIM, 1))

    def forward(self, features):
        return self.network((features-self.mean)/self.scale).squeeze(-1)
