"""移動ノズルから離れた滴の運動と、縮約容器モデルの適用条件。"""
import math
import numpy as np

LEGACY = 'stationary-v1'
MOVING_NOZZLE = 'moving-nozzle-v2'
GRAVITY_M_S2 = 9.81
TIME_EPSILON_S = 1e-9
STATE_EPSILON = 1e-12
MAX_TILT_RAD = math.radians(5.)
MAX_FLOW_SPEED_M_S = .002
DOMAIN_VERSION = 'quasistatic-downward-v1'
DOMAIN = dict(version=DOMAIN_VERSION,max_tilt_rad=MAX_TILT_RAD,max_flow_speed_m_s=MAX_FLOW_SPEED_M_S,
              empirically_validated=False,relative_jet_speed_m_s=0.,air_drag=False)


def nozzle_motion(position,center_of_mass,linear_velocity,angular_velocity,axis):
    velocity=np.asarray(linear_velocity)+np.cross(angular_velocity,np.asarray(position)-center_of_mass)
    return dict(position_m=list(position),velocity_m_s=velocity.tolist(),axis=list(axis),angular_rad_s=list(angular_velocity))


def emit_drop(ident,birth,start,end,before,after):
    if (not math.isfinite(birth) or not start<end or birth<start-TIME_EPSILON_S or birth>end+TIME_EPSILON_S):
        raise ValueError('滴発生時刻が物理区間外です')
    birth=min(end,max(start,birth));fraction=(birth-start)/(end-start)
    def interpolate(key):
        return ((1-fraction)*np.asarray(before[key])+fraction*np.asarray(after[key])).tolist()
    return dict(id=ident,position=interpolate('position_m'),velocity=interpolate('velocity_m_s'),emitted_at=birth)


def advance_drop(drop,start,end):
    start=max(start,drop['emitted_at']);duration=end-start
    if duration<=0:return None
    old=list(drop['position']);velocity=drop['velocity']
    drop['position']=[old[i]+velocity[i]*duration-(.5*GRAVITY_M_S2*duration**2 if i==2 else 0.) for i in range(3)]
    drop['velocity'][2]-=GRAVITY_M_S2*duration
    return dict(id=drop['id'],start_time=start,time=end,start=old,end=list(drop['position']))


def domain_sample(start,end,before,after,bottle_before,bottle_after,emitted):
    if not math.isfinite(start) or not math.isfinite(end) or end<=start:
        raise ValueError('物理適用条件の時刻が不正です')
    tilts=[];speeds=[];angular=[]
    for value in (before,after):
        for key in ('position_m','velocity_m_s','axis','angular_rad_s'):
            array=np.asarray(value[key])
            if array.shape!=(3,) or not np.isfinite(array).all():raise ValueError('ノズル観測が不正です: '+key)
        if not math.isclose(float(np.linalg.norm(value['axis'])),1.,rel_tol=0.,abs_tol=1e-6):
            raise ValueError('ノズル方向は単位ベクトルである必要があります')
        tilts.append(math.acos(float(np.clip(-value['axis'][2],-1.,1.))))
        speeds.append(float(np.linalg.norm(value['velocity_m_s'])))
        angular.append(float(np.linalg.norm(value['angular_rad_s'])))
    for snapshot in (bottle_before,bottle_after):
        if not all(math.isfinite(snapshot[key]) for key in ('displacement_mm','pressure_kpa','tip_ml','reservoir_ml')):
            raise ValueError('容器の採点記録が非有限です')
    active=bool(emitted) or any(snapshot[key]>STATE_EPSILON for snapshot in (bottle_before,bottle_after)
                               for key in ('displacement_mm','pressure_kpa','tip_ml'))
    flowing=bool(emitted) or bottle_before['reservoir_ml']-bottle_after['reservoir_ml']>STATE_EPSILON
    tilt_bound=max(tilts)+max(angular)*(end-start)
    violations=[]
    if active and tilt_bound>MAX_TILT_RAD:violations.append('bottle_tilt')
    if flowing and max(speeds)>MAX_FLOW_SPEED_M_S:violations.append('flow_nozzle_speed')
    return dict(start_time=start,time=end,active=active,flowing=flowing,tilt_bound_rad=tilt_bound,
                max_nozzle_speed_m_s=max(speeds),valid=not violations,violations=violations)


def domain_summary(samples,*,expected_intervals=None):
    complete=bool(samples) and (expected_intervals is None or len(samples)==expected_intervals)
    previous=0.
    for row in samples:
        complete &= abs(row['start_time']-previous)<=TIME_EPSILON_S and row['time']>row['start_time']
        previous=row['time']
    return dict(version=DOMAIN_VERSION,samples=len(samples),coverage_complete=bool(complete),
                valid=bool(complete) and all(row['valid'] for row in samples),
                violation_samples=sum(not row['valid'] for row in samples),
                max_tilt_bound_rad=max((row['tilt_bound_rad'] for row in samples if row['active']),default=None),
                max_flow_nozzle_speed_m_s=max((row['max_nozzle_speed_m_s'] for row in samples if row['flowing']),default=None))
