"""元の評価条件を保ち、追加した開発用ストレス例を自然開閉だけへ加える。"""
import math
import hashlib
from pathlib import Path

import numpy as np
from eyedrop.drop_kinematics import MOVING_NOZZLE, LEGACY

ARM_JOINTS=7
VISIBLE_HOME_YAW_RAD=math.pi/4
START_POSITION_TOLERANCE_M=1e-5
START_AXIS_TOLERANCE_RAD=math.radians(.1)
STRICT_OPEN_THRESHOLD=.95
LEGACY_OPEN_THRESHOLD=.9
SPECIFIED_SCORING_VERSION='first-surface-specified-eye-v3'
EXPECTED_EYE='left'


def prepare_downward_start(world,seed):
    """未使用の準備用worldだけで、ノズル位置を保つ下向き姿勢を解く。"""
    snapshot=world.bottle.snapshot()
    if world.time!=0. or world.drops or any(snapshot[k]!=0. for k in ('displacement_mm','pressure_kpa','tip_ml')):
        raise ValueError('開始姿勢の計算に実行中または液が残るworldを使いません')
    seed=np.asarray(seed,dtype=float)
    limits=np.asarray([(world.joint_info[i][8],world.joint_info[i][9]) for i in range(ARM_JOINTS)])
    if seed.shape!=(ARM_JOINTS,) or not np.isfinite(seed).all() or ((seed<limits[:,0])|(seed>limits[:,1])).any():
        raise ValueError('開始姿勢の関節が可動域外です')
    world.reset(seed.tolist())
    target=list(world.nozzle())
    solution=np.asarray(world.ik(target),dtype=float)
    if solution.shape!=(ARM_JOINTS,) or not np.isfinite(solution).all() or ((solution<limits[:,0])|(solution>limits[:,1])).any():
        raise ValueError('下向き開始姿勢のIK解が不正です')
    world.reset(solution.tolist())
    actual=world.nozzle_kinematics()
    position=np.asarray(actual['position_m']);axis=np.asarray(actual['axis'])
    if (position.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(axis).all()
            or not math.isclose(float(np.linalg.norm(axis)),1.,abs_tol=1e-6)):
        raise ValueError('開始姿勢の実測位置または方向が不正です')
    error=float(np.linalg.norm(position-target));tilt=math.acos(float(np.clip(-axis[2],-1.,1.)))
    if error>START_POSITION_TOLERANCE_M or tilt>START_AXIS_TOLERANCE_RAD:
        raise ValueError('下向き開始姿勢が位置または方向の許容差を満たしません')
    if world.contacts(swept=True):raise ValueError('下向き開始姿勢の接触余裕がありません')
    return solution.tolist(),dict(seed_joints=seed.tolist(),resolved_joints=solution.tolist(),target_nozzle_m=target,
        position_error_m=error,tilt_rad=tilt,position_tolerance_m=START_POSITION_TOLERANCE_M,
        axis_tolerance_rad=START_AXIS_TOLERANCE_RAD,scope='initialization-only;not-path-or-person-contact-proof')


def fixed_motion_task(condition,drops,hits,drop_model,domain):
    if drop_model==MOVING_NOZZLE:
        if domain is None or not domain['coverage_complete'] or not domain['valid']:return False
        if condition!='natural':return drops==0 and not hits
        if drops!=1 or len(hits)!=1:return False
        hit=hits[0]
        return (hit.get('result')=='target' and hit.get('open',0)>=STRICT_OPEN_THRESHOLD
                and hit.get('scoring_version')==SPECIFIED_SCORING_VERSION
                and hit.get('expected_eye')==EXPECTED_EYE and hit.get('hit_eye')==EXPECTED_EYE)
    if drop_model!=LEGACY:raise ValueError('未登録の滴モデルです')
    return (drops==1 and len(hits)==1 and hits[0]['result']=='target' and hits[0]['open']>=LEGACY_OPEN_THRESHOLD) if condition=='natural' else drops==0


def capture_loaded_sources(directory,modules):
    """記録先を旧runtimeへ固定せず、実際にimportしたソースを保全する。"""
    hashes={};locations={}
    for name,module in sorted(modules.items()):
        if (name!='eyedrop' and not name.startswith('eyedrop.')) or not getattr(module,'__file__',None):continue
        source=Path(module.__file__).resolve()
        if source.suffix!='.py':continue
        relative=Path('__init__.py') if name=='eyedrop' else Path(*name.split('.')[1:]).with_suffix('.py')
        destination=Path(directory)/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        content=source.read_bytes();destination.write_bytes(content)
        hashes[relative.as_posix()]=hashlib.sha256(content).hexdigest();locations[relative.as_posix()]=str(source)
    return dict(hashes=hashes,locations=locations)


def resolve_start_joints(mode,teacher):
    if len(teacher)!=ARM_JOINTS or not all(math.isfinite(q) for q in teacher):raise ValueError('元の開始関節が不正です')
    if mode=='teacher':return list(teacher)
    if mode=='visible-home':
        from eyedrop.panda_world import HOME
        return [HOME[0]-VISIBLE_HOME_YAW_RAD,*HOME[1:]]
    raise ValueError('開始姿勢が未登録です')
def evaluation_cases(protocol,condition,extra_ids):
    if len(extra_ids)!=len(set(extra_ids)):raise ValueError('追加する診断例が重複しています')
    training={row['id']:row for row in protocol['train']}
    if any(identifier not in training for identifier in extra_ids):raise ValueError('未知の追加診断例です')
    return protocol['validation']+[training[identifier] for identifier in extra_ids] if condition=='natural' else protocol['validation']
