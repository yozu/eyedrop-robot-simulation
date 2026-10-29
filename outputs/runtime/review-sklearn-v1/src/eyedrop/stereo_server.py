"""3枚のRGBと許可した自己状態だけで2眼方式を逐次評価する。"""
import base64
import hashlib
import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import torch
from .face_features import FaceFeatures
from .learning_env import LearningRun
from .state_control import Memory
from .ready_confirmation import ReadyConfirmation
from .open_image_features import eye_image_features, INPUT_COUNT
from .state_image_server import FACE_SHA, MOTION_SHA, DEPTH_SHA, MAX_FRAMES
from .stereo_control import StereoPosition, StereoController

ROOT = Path('outputs/learning')
OUT = ROOT / 'stereo-control-v1'
MAX_BASE64 = 4 * 1024 * 1024
CALIBRATION_SHA = '0c227f48a987e6233a2fb9760cbaa88f29c901091cef6571ef556ce5cb99a966'
app = FastAPI()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StereoReset(BaseModel):
    model_config = ConfigDict(extra='forbid')
    seed: int
    mode: Literal['open-calibrated']


class StereoStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str
    frame: int
    image: str = Field(max_length=MAX_BASE64)
    high_left: str = Field(max_length=MAX_BASE64)
    high_right: str = Field(max_length=MAX_BASE64)


class StereoSession:
    def __init__(self, face, controller, calibration, confirmation_frames):
        self.face = face
        self.controller = controller
        self.calibration = calibration
        self.confirmation_frames = confirmation_frames
        self.run = None
        self.run_id = None

    def reset(self, request):
        if self.run is not None:
            self.run.__exit__()
        self.run = LearningRun(request.seed).__enter__()
        self.run_id = str(uuid.uuid4())
        self.memory = Memory()
        self.confirmation = ReadyConfirmation(self.confirmation_frames)
        return {'run_id': self.run_id, 'sim': self.run.snapshot()}

    def step(self, request):
        if self.run is None or request.run_id != self.run_id or request.frame != self.run.frame:
            raise ValueError('試行IDまたはフレームが一致しません')
        if request.frame >= MAX_FRAMES:
            raise ValueError('試行フレーム上限です')
        started = time.monotonic()
        contents = [base64.b64decode(value, validate=True) for value in (request.image, request.high_left, request.high_right)]
        features = self.face.extract(contents[0]).cpu().numpy()
        opening_features = eye_image_features(contents[0], features)
        opened = self.confirmation.update(float(self.calibration.predict([opening_features])[0]))
        own = self.run.snapshot()['observation']['state']
        result = self.controller.action(features, own, self.memory, *contents, opened)
        sim = self.run.step(result['action'])
        self.memory.update(opened, result['action'][3])
        return {'sim': sim, **result, 'face_features': features.tolist(), 'opening_features': opening_features.tolist(),
                **{key: hashlib.sha256(content).hexdigest() for key, content in zip(('image_sha256', 'high_left_sha256', 'high_right_sha256'), contents)},
                'seconds': time.monotonic() - started}


@app.on_event('startup')
def startup():
    torch.set_num_threads(2)
    static = json.loads((OUT / 'report.json').read_bytes())
    assert static['static_gate']
    assert digest(OUT / 'captures.json') == static['captures_sha256']
    assert digest(OUT / 'geometry.json') == static['geometry_sha256']
    assert digest(ROOT / 'state-control-v1/revision-2/policy.pkl') == MOTION_SHA
    assert digest(ROOT / 'rgb-depth-v1/position.pkl') == DEPTH_SHA
    assert digest(ROOT / 'c-open-v7/calibration.pkl') == CALIBRATION_SHA
    summary = json.loads((ROOT / 'c-open-v7/rollouts/summary.json').read_bytes())
    assert digest(Path('preview/scene.js')) == summary['scene_sha256']
    captures = json.loads((OUT / 'captures.json').read_bytes())
    assert captures['scene_sha256'] == summary['scene_sha256']
    face = FaceFeatures(ROOT / 'separate-v2/openness', 'mps')
    assert face.sha256 == FACE_SHA
    position = StereoPosition(ROOT / 'target-readout-v1', ROOT / 'rgb-depth-v1/position.pkl')
    assert position.locator.sha256 == static['locator_sha256']
    motion = pickle.loads((ROOT / 'state-control-v1/revision-2/policy.pkl').read_bytes())
    calibration = pickle.loads((ROOT / 'c-open-v7/calibration.pkl').read_bytes())
    assert calibration.n_features_in_ == INPUT_COUNT
    frames = summary['health']['confirmation_frames']
    app.state.session = StereoSession(face, StereoController(position, motion), calibration, frames)
    source_names = ('stereo_server.py', 'stereo_control.py', 'stereo_geometry.py', 'target_supervision.py', 'state_control.py')
    sources = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    app.state.health = {'pid': os.getpid(), 'device': 'mps', 'motion_sha256': MOTION_SHA, 'face_sha256': FACE_SHA,
                        'baseline_position_sha256': DEPTH_SHA, 'locator_sha256': position.locator.sha256,
                        'calibration_sha256': CALIBRATION_SHA, 'confirmation_frames': frames,
                        'observation_contract': 'original-left256 plus calibrated left/right512 RGB; own state/history only',
                        'runtime_sources': sources, 'static_report_sha256': digest(OUT / 'report.json'),
                        'protocol_sha256': digest(Path('experiments/stereo-control-v1.json'))}


@app.get('/health')
def health():
    return app.state.health


@app.post('/reset')
def reset(request: StereoReset):
    return app.state.session.reset(request)


@app.post('/step')
def step(request: StereoStep):
    try:
        return app.state.session.step(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
