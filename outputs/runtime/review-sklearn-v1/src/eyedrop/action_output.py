"""学習時と推論時で同じ出力層を構成し、保存重みとの組を検証する。"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

OUTPUT_VERSION = 1
OUTPUT_MODES = ("identity", "tanh", "binary-pressure")
PRESSURE_AXIS = 3
BINARY_PRESSURE_MM = 1.0
PRESSURE_RANGE_MM = 3.0
PRESSURE_RELEASE_NORMALIZED = -1.0
PRESSURE_ON_NORMALIZED = 2 * BINARY_PRESSURE_MM / PRESSURE_RANGE_MM - 1
SPEC_FILE = "action-output.json"
WEIGHT_FILE = "model.safetensors"


class BoundedActionHead(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, values):
        return torch.tanh(F.linear(values, self.weight, self.bias))


class BinaryPressureHead(BoundedActionHead):
    def forward(self, values):
        logits=F.linear(values,self.weight,self.bias)
        if not torch.isfinite(logits).all():raise ValueError('押圧分類の値が非有限です')
        pressure=torch.where(logits[...,PRESSURE_AXIS:]>0,
                             PRESSURE_ON_NORMALIZED,PRESSURE_RELEASE_NORMALIZED)
        return torch.cat((torch.tanh(logits[...,:PRESSURE_AXIS]),pressure),dim=-1)


def configure_output(policy, mode):
    if mode not in OUTPUT_MODES:
        raise ValueError("未対応の出力構成です")
    if not isinstance(policy.model.action_head, nn.Linear):
        raise ValueError("出力層は未変換の線形層である必要があります")
    if mode == "tanh":
        policy.model.action_head = BoundedActionHead(policy.model.action_head)
    elif mode == "binary-pressure":
        policy.model.action_head = BinaryPressureHead(policy.model.action_head)


def save_output_spec(folder, mode):
    if mode not in OUTPUT_MODES:
        raise ValueError("未対応の出力構成です")
    folder = Path(folder)
    spec = {"version":OUTPUT_VERSION,"mode":mode,
            "model_sha256":hashlib.sha256((folder/WEIGHT_FILE).read_bytes()).hexdigest()}
    with (folder/SPEC_FILE).open("x",encoding="utf-8") as stream:
        json.dump(spec,stream,indent=2)


def load_output_spec(policy, folder):
    folder = Path(folder)
    path = folder/SPEC_FILE
    # 仕様ファイル導入前の保存済みモデルは線形出力で学習されている。
    if not path.exists():
        return "identity"
    spec = json.loads(path.read_bytes())
    if set(spec) != {"version","mode","model_sha256"} or spec["version"] != OUTPUT_VERSION:
        raise ValueError("出力仕様の形式または版が不正です")
    if spec["model_sha256"] != hashlib.sha256((folder/WEIGHT_FILE).read_bytes()).hexdigest():
        raise ValueError("出力仕様とモデル重みが一致しません")
    configure_output(policy,spec["mode"])
    return spec["mode"]
