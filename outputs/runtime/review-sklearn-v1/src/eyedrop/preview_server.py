"""ローカル限定の描画・物理接続診断サーバー。"""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4
import mimetypes
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .diagnostic import DiagnosticRun, DISPENSE_HEIGHT, MIN_DISPENSE_HEIGHT, APPROACH_HEIGHT
from . import learning_api

lock = Lock()
active = None
run_id = None


@asynccontextmanager
async def lifespan(app):
    global active, run_id
    try:
        yield
    finally:
        with lock:
            if active is not None:
                active.__exit__()
            active = None
            run_id = None
        with learning_api.lock:
            learning_api.close()


app = FastAPI(lifespan=lifespan)
app.include_router(learning_api.router)


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    eye: tuple[float, float, float]
    dispense_height: float = Field(default=DISPENSE_HEIGHT,ge=MIN_DISPENSE_HEIGHT,le=APPROACH_HEIGHT)


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: str
    force_closed: bool = False


@app.post("/api/start")
def start(body: Start):
    global active, run_id
    with lock:
        if active is not None:
            active.__exit__()
        active = DiagnosticRun(body.eye,body.dispense_height).__enter__()
        run_id = str(uuid4())
        return {"run": run_id, **active.snapshot()}


@app.post("/api/step")
def step(body: Step):
    with lock:
        if active is None or body.run != run_id:
            raise HTTPException(409, "別の実行に切り替わりました。開始し直してください")
        return active.advance(body.force_closed)


class BottleTrialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target_mm: float = Field(ge=0, le=3)
    hold_s: float = Field(ge=.1, le=5)
    press_speed_mm_s: float = Field(default=2., ge=.1, le=10)
    release_speed_mm_s: float = Field(default=2., ge=.1, le=10)


@app.post("/api/bottle/trial")
def bottle_trial(body: BottleTrialRequest):
    from .bottle_trial import trial
    return trial(**body.model_dump())


PREVIEW = Path(__file__).resolve().parents[2] / "preview"
# Windowsの拡張子登録に左右されず、ブラウザーがモジュールを読み込めるようにする。
mimetypes.init()
mimetypes.add_type("text/javascript", ".mjs")
app.mount("/", StaticFiles(directory=PREVIEW, html=True), name="preview")
