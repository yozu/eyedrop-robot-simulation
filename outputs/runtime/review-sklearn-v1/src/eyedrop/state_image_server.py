"""画像と自己状態だけで固定方策を実行する、Macの逐次評価用サーバー。"""
import base64,hashlib,pickle,time,uuid,os,json
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict
from .face_features import FaceFeatures
from .learning_env import LearningRun
from .perception_control import PerceptionControl
from .state_control import Memory
from .open_image_features import eye_image_features,INPUT_COUNT
from .face_features import FEATURE_COUNT
from .ready_confirmation import ReadyConfirmation

ROOT=Path('outputs/learning')
MOTION_SHA='62b6e5288a800653747d5199922b94017193f9e4750f149623e0f4eb9c8b2dd8'
FACE_SHA='7e4c8716130ed4323ab4e6b7ca7e1841d0fb13c4e5f0e89e595b6fe1ac1121a2'
DEPTH_SHA='af7875d1661484823711bf601f68259fe36aca1293abafd604ed69503d2257c1'
MAX_FRAMES=300
app=FastAPI()

def validated_position(folder):
    report=json.loads((folder/'report.json').read_bytes());path=folder/'position.pkl'
    if not (report['static_gate'] and report['geometry_gate'] and report['face_sha256']==FACE_SHA
            and report['position_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('位置モデルの検証結果またはハッシュが一致しません')
    return path

class ResetInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    seed:int
    mode:Literal['baseline','rgb-depth','open-calibrated']

class StepInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    run_id:str
    frame:int
    image:str

class Session:
    def __init__(self,face,adapters,calibration=None,confirmation_frames=1):
        self.face=face;self.adapters=adapters;self.calibration=calibration;self.run=None;self.run_id=None
        self.confirmation_frames=confirmation_frames
    def reset(self,request):
        if request.mode not in self.adapters:raise ValueError('指定された評価モデルは読み込まれていません')
        if self.run is not None:self.run.__exit__()
        self.run=LearningRun(request.seed).__enter__();self.run_id=str(uuid.uuid4())
        self.mode=request.mode;self.memory=Memory()
        self.ready_confirmation=ReadyConfirmation(self.confirmation_frames)
        return {'run_id':self.run_id,'sim':self.run.snapshot()}
    def step(self,request):
        if self.run is None or request.run_id!=self.run_id or request.frame!=self.run.frame:
            raise ValueError('試行IDまたはフレームが一致しません')
        if request.frame>=MAX_FRAMES:raise ValueError('試行フレーム上限です')
        content=base64.b64decode(request.image,validate=True);start=time.monotonic()
        features=self.face.extract(content).cpu().tolist()
        own=self.run.snapshot()['observation']['state']
        opening_features=features;calibrated=None
        if self.mode=='open-calibrated':
            if self.calibration.n_features_in_==INPUT_COUNT:opening_features=eye_image_features(content,features).tolist()
            elif self.calibration.n_features_in_!=FEATURE_COUNT:raise ValueError('開眼校正の入力形式が不正です')
            calibrated=float(self.calibration.predict([opening_features])[0])
            calibrated=self.ready_confirmation.update(calibrated)
        result=self.adapters[self.mode].action(features,own,self.memory,content if self.mode in ('rgb-depth','open-calibrated') else None,calibrated_openness=calibrated)
        sim=self.run.step(result['action']);self.memory.update(result['predicted_openness'],result['action'][3])
        return {'sim':sim,**result,'face_features':features,'opening_features':opening_features,'image_sha256':hashlib.sha256(content).hexdigest(),'seconds':time.monotonic()-start}

@app.on_event('startup')
def startup():
    import torch
    torch.set_num_threads(2)
    content=(ROOT/'state-control-v1/revision-2/policy.pkl').read_bytes()
    assert hashlib.sha256(content).hexdigest()==MOTION_SHA
    motion=pickle.loads(content);face=FaceFeatures(ROOT/'separate-v2/openness','mps');assert face.sha256==FACE_SHA
    positions={'baseline':ROOT/'state-control-v1/perception-connection/position.pkl','rgb-depth':ROOT/'rgb-depth-v1/position.pkl'}
    assert hashlib.sha256(positions['rgb-depth'].read_bytes()).hexdigest()==DEPTH_SHA
    calibration=None;confirmation_frames=1;app.state.calibration_sha=None
    if os.environ.get('EYEDROP_OPEN_CALIBRATION'):
        folder=Path(os.environ['EYEDROP_OPEN_CALIBRATION']);report=json.loads((folder/'report.json').read_bytes())
        content=(folder/'calibration.pkl').read_bytes();app.state.calibration_sha=hashlib.sha256(content).hexdigest()
        assert report['static_gate'] and report['face_sha256']==FACE_SHA and report['calibration_sha256']==app.state.calibration_sha
        calibration=pickle.loads(content);positions['open-calibrated']=positions['rgb-depth']
        confirmation_frames=report.get('confirmation_frames',1)
    if os.environ.get('EYEDROP_POSITION'):
        if calibration is None:raise ValueError('位置補正には固定した開眼モデルの指定が必要です')
        positions['open-calibrated']=validated_position(Path(os.environ['EYEDROP_POSITION']))
    app.state.hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in positions.items()}
    app.state.confirmation_frames=confirmation_frames
    app.state.session=Session(face,{k:PerceptionControl(pickle.loads(p.read_bytes()),motion) for k,p in positions.items()},calibration,confirmation_frames)

@app.get('/health')
def health():return {'pid':os.getpid(),'device':'mps','motion_sha256':MOTION_SHA,'face_sha256':FACE_SHA,'position_sha256':app.state.hashes,'calibration_sha256':app.state.calibration_sha,'confirmation_frames':app.state.confirmation_frames}

@app.post('/reset')
def reset(request:ResetInput):return app.state.session.reset(request)

@app.post('/step')
def step(request:StepInput):
    try:return app.state.session.step(request)
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
