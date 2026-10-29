"""画像＋自己状態から4軸指令を出すACT。正解座標は受け取らない。"""
import io
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from .face_features import FEATURE_COUNT, CACHE_VERSION
from .evaluation_reservation import require_development_person
from .observation_constants import (
    HISTORY_STEPS, MAX_EPISODE_HISTORY, HISTORY_STATE_SIZE, PRESSURE_MEMORY_FEATURES)

IMAGE_KEY = "observation.images.camera"
STATE_KEY = "observation.state"
SIDE = 128
CHUNK = 8
LOW = torch.tensor([-.08,-.16,.02,0.])
HIGH = torch.tensor([.08,.12,.35,3.])
PREPROCESS_VERSION = "rgb-state-mechanical-range-v1"


def normalize(values):
    return 2*(values-LOW.to(values.device))/(HIGH-LOW).to(values.device)-1


def denormalize(values):
    return (values+1)/2*(HIGH-LOW).to(values.device)+LOW.to(values.device)


def image_tensor(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.mode=="RGBA" and image.getchannel("A").getextrema()==(255,255):
            image=image.convert("RGB")
        if image.mode!="RGB" or image.size!=(256,256):
            raise ValueError("256×256 RGB画像が必要です")
        pixels=np.asarray(image.resize((SIDE,SIDE),Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(pixels).permute(2,0,1).float()/255.


def make_policy(device, use_vae=True, history_steps=1, face_features=0, pressure_memory=False):
    if face_features not in (0,FEATURE_COUNT):
        raise ValueError("顔特徴の次元が不正です")
    config=ACTConfig(
        device=device,push_to_hub=False,
        input_features={IMAGE_KEY:PolicyFeature(FeatureType.VISUAL,(3,SIDE,SIDE)),
                        STATE_KEY:PolicyFeature(FeatureType.STATE,(4*history_steps+face_features+
                            (PRESSURE_MEMORY_FEATURES if pressure_memory else 0),))},
        output_features={"action":PolicyFeature(FeatureType.ACTION,(4,))},
        chunk_size=CHUNK,n_action_steps=1,pretrained_backbone_weights=None,
        dim_model=128,n_heads=4,dim_feedforward=512,n_encoder_layers=2,
        n_decoder_layers=1,n_vae_encoder_layers=2,latent_dim=16,kl_weight=.1,
        use_vae=use_vae)
    return ACTPolicy(config).to(device)


class Episodes(Dataset):
    def __init__(self,root,split,history_steps=1,face_cache=None,pressure_memory=False):
        self.history_steps=history_steps
        if pressure_memory and history_steps!=HISTORY_STEPS:
            raise ValueError("押圧履歴は8時点履歴と併用してください")
        self.pressure_memory=pressure_memory
        self.face_cache=json.loads(Path(face_cache).read_bytes()) if face_cache else None
        if self.face_cache and self.face_cache["version"]!=CACHE_VERSION:
            raise ValueError("顔特徴キャッシュの版が異なります")
        self.face_count=FEATURE_COUNT if self.face_cache else 0
        self.faces=[]
        self.pressure_peaks=[]
        protocol_path=Path(root)/"protocol.json"
        protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
        digest=hashlib.sha256()
        self.episodes=[]
        self.samples=[]
        for folder in sorted(Path(root).glob("episode-*")):
            meta=json.loads((folder/"metadata.json").read_text())
            require_development_person(meta.get("person"), meta.get("seed"))
            if meta["split"]!=split:
                continue
            if protocol and (split not in ("train","validation") or
                             meta.get("seed") not in protocol[f"{split}_seeds"] or
                             meta.get("seed") in protocol["test_seeds"]):
                raise ValueError("固定した分割に一致しない実演です")
            if meta["contacts"]:
                raise ValueError(f"接触した実演は学習に使用しません: {folder}")
            if meta.get("accepted") is False:
                raise ValueError(f"不採用の教師実演です: {folder}")
            rows=json.loads((folder/"steps.json").read_text())
            digest.update(json.dumps({"metadata":meta,"steps":rows},sort_keys=True).encode())
            frames=[]
            face_rows=[]
            for row in rows:
                if set(row)!={"image","state","action"}:
                    raise ValueError("学習列に未定義項目があります")
                state=torch.tensor(row["state"],dtype=torch.float32)
                action=torch.tensor(row["action"],dtype=torch.float32)
                if state.shape!=(4,) or action.shape!=(4,) or not torch.isfinite(torch.cat([state,action])).all():
                    raise ValueError("状態・操作が不正です")
                image_path=(folder/row["image"]).resolve()
                if not image_path.is_relative_to(folder.resolve()):
                    raise ValueError("画像がエピソードの外にあります")
                content=image_path.read_bytes()
                digest.update(content)
                if self.face_cache:
                    key=folder.name+"/"+row["image"]
                    entry=self.face_cache["entries"][key]
                    if hashlib.sha256(content).hexdigest()!=entry["image_sha256"]:
                        raise ValueError("顔特徴と画像のハッシュが一致しません")
                    features=torch.tensor(entry["features"],dtype=torch.float32)
                    if features.shape!=(FEATURE_COUNT,) or not torch.isfinite(features).all() or (features.abs()>1).any():
                        raise ValueError("顔特徴が不正です")
                    face_rows.append(features)
                frames.append((image_tensor(content),normalize(state),normalize(action)))
            index=len(self.episodes)
            self.episodes.append(frames)
            self.faces.append(face_rows)
            if self.pressure_memory:
                # 各時点までの自己押圧最大値を一度だけ計算する。未来の値は使わない。
                self.pressure_peaks.append(torch.stack([row[1][3] for row in frames]).cummax(0).values)
            self.samples.extend((index,i) for i in range(len(frames)))
        if not self.samples:
            raise ValueError(f"{split} の実演がありません")
        self.sha256=digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        ep,frame=self.samples[index]
        rows=self.episodes[ep]
        image,state,_=rows[frame]
        if self.history_steps>1:
            state=torch.cat([rows[max(0,frame-i)][1]
                             for i in reversed(range(self.history_steps))])
        if self.pressure_memory:
            peak=self.pressure_peaks[ep][frame]
            state=torch.cat((state,peak.reshape(1)))
        if self.face_cache:
            state=torch.cat((state,self.faces[ep][frame]))
        ids=[min(frame+i,len(rows)-1) for i in range(CHUNK)]
        return {IMAGE_KEY:image,STATE_KEY:state,
                "action":torch.stack([rows[i][2] for i in ids]),
                "action_is_pad":torch.tensor([frame+i>=len(rows) for i in range(CHUNK)])}
