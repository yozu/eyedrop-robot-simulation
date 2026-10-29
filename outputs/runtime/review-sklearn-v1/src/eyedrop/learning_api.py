from threading import Lock
from time import monotonic
from uuid import uuid4
import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .learning_env import LearningRun
from .observation_constants import MAX_EPISODE_HISTORY, HISTORY_STEPS

router = APIRouter(prefix="/api/learning")
lock = Lock()
active = None
run_id = None
MAX_INFERENCE_AGE_S = 2.


class Reset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(ge=0)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run: str
    frame: int = Field(ge=0)
    action: tuple[float,float,float,float]


class PolicyObservation(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    run:str
    frame:int=Field(ge=0)
    state:tuple[float,float,float,float]
    image:str=Field(max_length=512*1024)
    history:list[tuple[float,float,float,float]]|None=Field(default=None,min_length=HISTORY_STEPS,max_length=MAX_EPISODE_HISTORY)


@router.post("/infer")
def infer(body:PolicyObservation):
    with lock:
        require_current(body.run,body.frame)
    started=monotonic()
    url=os.environ.get("EYEDROP_ACTION_URL","http://127.0.0.1:18762/act")
    try:
        reply=httpx.post(url,json=body.model_dump(exclude={"run","frame"}),timeout=MAX_INFERENCE_AGE_S)
        reply.raise_for_status()
        with lock:
            require_current(body.run,body.frame)
        if monotonic()-started>MAX_INFERENCE_AGE_S:
            raise HTTPException(504,"推論応答が古いため操作しません")
        return reply.json()
    except httpx.HTTPError as error:
        raise HTTPException(503,"学習済み方策サーバーから応答がありません") from error


def require_current(run,frame):
    if active is None or run!=run_id or frame!=active.frame:
        raise HTTPException(409,"実行または観測が更新されています")


def close():
    global active
    if active is not None:
        active.__exit__()
        active = None


@router.post("/reset")
def reset(body: Reset):
    global active, run_id
    with lock:
        close()
        active = LearningRun(body.seed).__enter__()
        run_id = str(uuid4())
        return {"run":run_id,**active.snapshot()}


@router.post("/step")
def step(body: Action):
    with lock:
        require_current(body.run,body.frame)
        try:
            return active.step(body.action)
        except ValueError as error:
            raise HTTPException(422,str(error)) from error
