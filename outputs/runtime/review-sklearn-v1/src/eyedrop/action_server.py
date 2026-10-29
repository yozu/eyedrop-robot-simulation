"""学習済みACT専用。画像と4軸の自己状態以外の観測を受け付けない。"""
import base64
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .action_policy import ACTPolicy, ACTConfig, IMAGE_KEY, STATE_KEY, LOW, HIGH, HISTORY_STEPS, normalize, denormalize, image_tensor
from .face_features import FaceFeatures, FEATURE_COUNT
from .action_output import load_output_spec
from .action_policy import MAX_EPISODE_HISTORY, HISTORY_STATE_SIZE, PRESSURE_MEMORY_FEATURES

MAX_IMAGE_BASE64=512*1024
STATE_TOLERANCE=1e-4
policy=None
model_hash=None
face_encoder=None
action_output="identity"
device=os.environ.get("EYEDROP_POLICY_DEVICE","cuda")


class PolicyInput(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    state:tuple[float,float,float,float]
    image:str=Field(max_length=MAX_IMAGE_BASE64)
    history:list[tuple[float,float,float,float]]|None=Field(default=None,min_length=HISTORY_STEPS,max_length=MAX_EPISODE_HISTORY)


@asynccontextmanager
async def lifespan(app):
    global policy,model_hash,face_encoder,action_output
    folder=Path(os.environ["EYEDROP_POLICY_PATH"])
    config=ACTConfig.from_pretrained(folder)
    config.device=device
    policy=ACTPolicy.from_pretrained(folder,config=config).to(device).eval()
    action_output=load_output_spec(policy,folder)
    policy.eval()
    model_hash=hashlib.sha256((folder/"model.safetensors").read_bytes()).hexdigest()
    face_encoder=None
    if config.input_features[STATE_KEY].shape[0] in (
        HISTORY_STATE_SIZE+FEATURE_COUNT,
        HISTORY_STATE_SIZE+FEATURE_COUNT+PRESSURE_MEMORY_FEATURES):
        face_encoder=FaceFeatures(folder/"face_model",device)
        reference=json.loads((folder/"face-model.json").read_bytes())
        if reference["sha256"]!=face_encoder.sha256:
            raise ValueError("操作学習時と顔予測モデルが一致しません")
    yield
    policy=None
    face_encoder=None


app=FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"mode":"learned-act","device":device,"model_sha256":model_hash,
            "inputs":["image","state","history"],"action_steps":1,
            "action_output":action_output,
            "history_steps":(policy.config.input_features[STATE_KEY].shape[0]-
                             (FEATURE_COUNT if face_encoder else 0))//4,
            "face_model_sha256":face_encoder.sha256 if face_encoder else None}


@app.post("/act")
def act(body:PolicyInput):
    if policy is None:
        raise HTTPException(503,"学習済み方策を読み込んでいません")
    started=time.perf_counter()
    try:
        content=base64.b64decode(body.image,validate=True)
        image=image_tensor(content).unsqueeze(0).to(device)
    except Exception as exc:
        raise HTTPException(422,"カメラ画像が不正です") from exc
    state=torch.tensor(body.state,dtype=torch.float32)
    if not ((state>=LOW-STATE_TOLERANCE)&(state<=HIGH+STATE_TOLERANCE)).all():
        raise HTTPException(422,"自己状態が可動範囲外です")
    state_size=policy.config.input_features[STATE_KEY].shape[0]
    with_memory=state_size in (HISTORY_STATE_SIZE+PRESSURE_MEMORY_FEATURES,
                              HISTORY_STATE_SIZE+FEATURE_COUNT+PRESSURE_MEMORY_FEATURES)
    with_face=state_size in (HISTORY_STATE_SIZE+FEATURE_COUNT,
                            HISTORY_STATE_SIZE+FEATURE_COUNT+PRESSURE_MEMORY_FEATURES)
    if state_size==HISTORY_STATE_SIZE or with_face or with_memory:
        if body.history is None:
            raise HTTPException(422,"このモデルには自己状態の履歴が必要です")
        history=torch.tensor(body.history,dtype=torch.float32)
        if not torch.equal(history[-1],state) or not ((history>=LOW-STATE_TOLERANCE)&(history<=HIGH+STATE_TOLERANCE)).all():
            raise HTTPException(422,"自己状態の履歴が不正です")
        normalized_history=normalize(history)
        state_input=normalized_history[-HISTORY_STEPS:].flatten()
        if with_memory:
            state_input=torch.cat((state_input,normalized_history[:,3].max().reshape(1)))
    elif state_size==4:
        state_input=normalize(state)
    else:
        raise HTTPException(500,"未対応のモデル入力です")
    predicted_face=None
    if with_face:
        if face_encoder is None:
            raise HTTPException(500,"顔予測モデルがありません")
        predicted_face=face_encoder.extract(content)
        state_input=torch.cat((state_input.to(device),predicted_face))
    with torch.inference_mode():
        chunk=policy.predict_action_chunk({IMAGE_KEY:image,STATE_KEY:state_input.unsqueeze(0).to(device)})
        raw=denormalize(chunk[0,0]).cpu()
    if not torch.isfinite(raw).all():
        raise HTTPException(500,"操作に非有限値が含まれています")
    bounded=torch.maximum(LOW,torch.minimum(HIGH,raw))
    return {"action":bounded.tolist(),"raw_action":raw.tolist(),"intervention":bool((raw!=bounded).any()),
            "pressure_memory":with_memory,
            "model_sha256":model_hash,
            "face_model_sha256":face_encoder.sha256 if with_face else None,
            "predicted_face_features":predicted_face.cpu().tolist() if with_face else None,
            "service_ms":(time.perf_counter()-started)*1000}
