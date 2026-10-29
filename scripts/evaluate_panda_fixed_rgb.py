"""画像による位置合わせと固定押圧を一連の物理動作で評価する。"""
import argparse
import base64
import json
import hashlib
from pathlib import Path
import time
from io import BytesIO
import sys

import numpy as np
from panda_rpc import PandaBrowser
from pressure_interval_audit import audit_pressure_interval
from panda_eval_cases import (evaluation_cases,resolve_start_joints,prepare_downward_start,fixed_motion_task,
                             capture_loaded_sources,STRICT_OPEN_THRESHOLD,SPECIFIED_SCORING_VERSION,EXPECTED_EYE)
from resumable_job import atomic_json
from eyedrop.panda_fixed_control import FixedEyeDrop
from eyedrop.blink_position_guard import BlinkPositionGuard
from eyedrop.panda_rgb import PandaRgb
from eyedrop.model_integrity import verified_model_bytes, restore_sklearn_model
from eyedrop.panda_world import PandaWorld, to_physics, ACTION_HZ, PHYSICS_HZ, BOTTLE_DEPTH_M,PHYSICS_JOINT_VELOCITY_CAP,synchronized_joint_target,ARM_JOINT_COUNT
from eyedrop.camera_position import CAMERA
from panda_trace_quality import SUPPORTED_FRAMES
from eyedrop.drop_kinematics import LEGACY,MOVING_NOZZLE,DOMAIN,domain_summary
from synchronized_demo import SynchronizedDemo

TARGET_HEIGHT_M = .05
ALIGN_M = .001
PATH_SPACING_RAD = .001
WAYPOINT_TOLERANCE_RAD = .001
PRESS_TOLERANCE_M = .0001
FRAMES = 300
SCORING_OPEN_THRESHOLD = .9
AUDIT_RAY_EXTENSION_M=.1
MM_PER_M=1000
PRESSURE_EPSILON_KPA=1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--oracle-diagnostic', action='store_true')
    parser.add_argument('--natural-only', action='store_true')
    parser.add_argument('--position-mode', choices=('legacy', 'full-stereo', 'peak-stereo'), default='legacy')
    parser.add_argument('--interrupt-after-press', type=int)
    parser.add_argument('--native-face', action='store_true')
    parser.add_argument('--locator-candidate')
    parser.add_argument('--locator-input', choices=('shared-face','resized'),default='shared-face')
    parser.add_argument('--bottle-input', choices=('command','measured'),default='command')
    parser.add_argument('--finger-release-speed',type=float,default=.002)
    parser.add_argument('--finger-press-speed',type=float,default=.002)
    parser.add_argument('--press-steps-limit',type=int)
    parser.add_argument('--standby-gap',type=float)
    parser.add_argument('--blink-position-hold',action='store_true')
    parser.add_argument('--closed-position-head')
    parser.add_argument('--all-position-head')
    parser.add_argument('--diagnostic-ready-head')
    parser.add_argument('--diagnostic-extra-cases',nargs='*',default=[])
    parser.add_argument('--person',default='original')
    parser.add_argument('--affine-blink-envelope',action='store_true')
    parser.add_argument('--start-pose',choices=('teacher','visible-home','visible-downward'),default='teacher')
    parser.add_argument('--drop-model',choices=(LEGACY,MOVING_NOZZLE),default=LEGACY)
    parser.add_argument('--audit-motion',action='store_true')
    parser.add_argument('--frames',type=int,choices=SUPPORTED_FRAMES,default=FRAMES)
    parser.add_argument('--joint-velocity-cap',type=float,default=PHYSICS_JOINT_VELOCITY_CAP)
    parser.add_argument('--continue-approach-after-first-open',action='store_true')
    parser.add_argument('--skip-natural-validation',action='store_true')
    parser.add_argument('--synchronize-arm-targets',action='store_true')
    parser.add_argument('--audit-target-v2',action='store_true')
    parser.add_argument('--guard-prepare-contact',action='store_true')
    parser.add_argument('--rgb-reference-head')
    parser.add_argument('--closed-half-only',action='store_true')
    parser.add_argument('--record-demonstration', action='store_true')
    args = parser.parse_args()
    moving=args.drop_model==MOVING_NOZZLE
    if args.audit_motion!=moving:
        raise ValueError('新しい滴モデルと全区間の運動記録は同時に明示してください')
    if args.start_pose=='visible-downward' and not moving:
        raise ValueError('新開始姿勢は新しい滴モデルで比較してください')
    open_threshold=STRICT_OPEN_THRESHOLD if moving else SCORING_OPEN_THRESHOLD
    if args.closed_half_only and (args.natural_only or args.interrupt_after_press or args.diagnostic_extra_cases or args.skip_natural_validation):
        raise ValueError('閉眼/半開眼だけの再試験と自然条件の指定を混在させません')
    if args.rgb_reference_head and (args.oracle_diagnostic or not args.native_face or args.position_mode!='full-stereo' or args.blink_position_hold):
        raise ValueError('画像再照合の診断にはnative/full-stereoが必要です。正解診断や旧位置保持と混在させません')
    if args.skip_natural_validation and not args.diagnostic_extra_cases:raise ValueError('追加自然条件なしで元の自然条件を省略しません')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    data = root/'outputs/learning/panda-cartesian-data-02/natural'
    protocol = json.loads((data/'protocol.json').read_text())
    initial = json.loads((data/'train-02/result.json').read_text())['initial_joints']
    initial=resolve_start_joints('visible-home' if args.start_pose=='visible-downward' else args.start_pose,initial)
    rest_gap = json.loads((data/'train-02/frames.json').read_text())[-1]['action'][7]
    dynamics={'bottle_input':args.bottle_input,'finger_release_speed_m_s':args.finger_release_speed,
              'finger_press_speed_m_s':args.finger_press_speed,'joint_velocity_cap':args.joint_velocity_cap,
              'drop_model':args.drop_model}
    start_audit=None
    if args.start_pose=='visible-downward':
        setup=PandaWorld(output,rest_gap=rest_gap,**dynamics)
        try:
            initial,start_audit=prepare_downward_start(setup,initial)
        finally:
            setup.close()
    if args.oracle_diagnostic and (args.locator_candidate or args.closed_position_head or args.all_position_head or args.diagnostic_ready_head):
        raise ValueError('正解診断と画像候補の評価を混在させません')
    model = None if args.oracle_diagnostic else PandaRgb(root/'outputs/learning', position_mode=args.position_mode,
                                                       locator_candidate=args.locator_candidate,locator_input=args.locator_input,closed_position_head=args.closed_position_head,all_position_head=args.all_position_head,diagnostic_ready_head=args.diagnostic_ready_head)
    if args.rgb_reference_head:
        import torch
        from PIL import Image
        from image_motion_validity import ObservedReferencePosition
        from eyedrop.panda_rgb import predict_rgb_reference
        folder=Path(args.rgb_reference_head);checkpoint=folder/'paired_resized.pkl'
        card=json.loads((folder/'paired_resized-model.json').read_bytes())
        if model.face.sha256!=card['encoder_sha256']:
            raise ValueError('固定した左右モデルまたはencoderが変更されています')
        content=verified_model_bytes(checkpoint,card['sha256'])
        reference_regressor=restore_sklearn_model(content)
        model.hashes['diagnostic_reference/paired_resized.pkl']=card['sha256']
        def predict_reference(images):
            return predict_rgb_reference(model.face,reference_regressor,images)
    atomic_json(output/'protocol.json', {'oracle_diagnostic': args.oracle_diagnostic,
                'person':args.person,
                'affine_blink_envelope':args.affine_blink_envelope,
                'start_pose':args.start_pose,
                'start_pose_audit':start_audit,'drop_model':args.drop_model,'audit_motion':args.audit_motion,
                'drop_domain':DOMAIN if moving else None,'open_threshold':open_threshold,
                'scoring_version':SPECIFIED_SCORING_VERSION if moving else None,
                'expected_eye':EXPECTED_EYE if moving else None,
                'frames':args.frames,'joint_velocity_cap':args.joint_velocity_cap,'continue_approach_after_first_open':args.continue_approach_after_first_open,
                'skip_natural_validation':args.skip_natural_validation,
                'synchronize_arm_targets':args.synchronize_arm_targets,
                'audit_target_v2':args.audit_target_v2,
                'guard_prepare_contact':args.guard_prepare_contact,
                'closed_half_only':args.closed_half_only,
                'rgb_reference_head':args.rgb_reference_head,'diagnostic_reference_horizon_s':args.frames/ACTION_HZ if args.rgb_reference_head else None,
                'training_updates': 0, 'initial_joints': initial, 'rest_gap': rest_gap,
                'record_demonstration':args.record_demonstration,
                'model_hashes': None if model is None else model.hashes,
                'position_mode': args.position_mode, 'native_face': args.native_face,
                'locator_candidate': args.locator_candidate,
                'locator_input': args.locator_input,
                'bottle_input':args.bottle_input,'finger_release_speed_m_s':args.finger_release_speed,'finger_press_speed_m_s':args.finger_press_speed,
                'press_steps_limit':args.press_steps_limit,'standby_gap':args.standby_gap,
                'blink_position_hold':args.blink_position_hold,'closed_position_head':args.closed_position_head,
                'all_position_head':args.all_position_head,'diagnostic_extra_cases':args.diagnostic_extra_cases,
                'diagnostic_ready_head':args.diagnostic_ready_head,
                'rest_gap_source': 'train-02 teacher final gap command; no validation outcome used',
                'scope': 'True eye/open used for control ONLY in explicitly labeled oracle diagnostic.'})
    sources = {}
    source_dir=output/'source';source_dir.mkdir()
    for source in [Path(__file__),root/'scripts/panda_rpc.py',root/'scripts/panda_browser.cjs',root/'scripts/panda_eval_cases.py',
                   root/'scripts/panda_trace_quality.py',root/'scripts/image_motion_validity.py',
                   root/'preview/panda-runtime.js',root/'preview/scene.js',root/'preview/camera-visibility.mjs',
                   root/'preview/panda-hybrid-rgb.js',root/'preview/anatomy-score.mjs',root/'preview/eye-target.js',
                   root/'scripts/pressure_interval_audit.py',root/'scripts/synchronized_demo.py']:
        content=source.read_bytes();(source_dir/source.name).write_bytes(content)
        sources[source.name]=hashlib.sha256(content).hexdigest()
    loaded=capture_loaded_sources(source_dir,dict(sys.modules));sources.update(loaded['hashes'])
    atomic_json(source_dir/'locations.json',loaded['locations'])
    atomic_json(source_dir/'hashes.json',sources)
    browser = PandaBrowser(output/'browser.log',person=args.person,hybrid_rgb=args.native_face)
    setup = PandaWorld(output, rest_gap=rest_gap,**dynamics)
    browser.call('setup', shapes=setup.drawing())
    setup.close()
    results = []
    started = time.monotonic()
    try:
        for condition in (('closed','half-open') if args.closed_half_only else ('natural',) if args.natural_only or args.interrupt_after_press else ('natural', 'closed', 'half-open')):
            for case in evaluation_cases(protocol,condition,args.diagnostic_extra_cases):
                if args.skip_natural_validation and condition=='natural' and case['id'] in {c['id'] for c in protocol['validation']}:continue
                directory = output/f'{condition}-{case["id"]}'
                directory.mkdir()
                reset=browser.call('reset', depth=case['depth_m'], phase=case['blink_phase_s'])
                if moving and (reset.get('scoring',{}).get('version')!=SPECIFIED_SCORING_VERSION
                               or reset['scoring'].get('expected_eye')!=EXPECTED_EYE):
                    raise ValueError('指定眼の採点契約が一致しません')
                geometry = browser.call('geometry')
                scoring_eye=None
                if args.audit_target_v2:
                    proxy=np.array(browser.call('teacherV2')['proxy']);direction=proxy-CAMERA;direction/=np.linalg.norm(direction)
                    hit=browser.call('score',start=CAMERA.tolist(),end=(proxy+AUDIT_RAY_EXTENSION_M*direction).tolist(),open=1.)
                    if hit is None or hit['result']!='target':raise ValueError('採点専用の眼表面を取得できません')
                    scoring_eye=hit['point']
                    atomic_json(directory/'scoring-target.json',{'eye_scene':scoring_eye,'scope':'teacherV2 surface for scoring only,never model/control input'})
                world = PandaWorld(directory, rest_gap=rest_gap,**dynamics)
                world.reset(initial)
                if args.affine_blink_envelope:
                    world.add_affine_geometry(geometry,browser.call('blinkGeometry'))
                else:
                    world.add_geometry(geometry)
                control = FixedEyeDrop(rest_gap=rest_gap,press_steps_limit=args.press_steps_limit,standby_gap=args.standby_gap,continue_approach_after_first_open=args.continue_approach_after_first_open,guard_prepare_contact=args.guard_prepare_contact)
                position_guard=BlinkPositionGuard() if args.blink_position_hold else None
                reference_guard=ObservedReferencePosition(predict_reference,max_age_seconds=args.frames/ACTION_HZ) if args.rgb_reference_head else None
                start_nozzle = np.array(world.nozzle())
                path = [list(initial)]
                held = None
                rows, contacts, hits, errors = [], [], [], []
                pressure_samples = {}
                motion_samples,domain_samples=[],[]
                closure_step = None
                recorder = SynchronizedDemo(world.own_state(), rest_gap) if args.record_demonstration else None
                demonstration = []
                try:
                    for step in range(args.frames):
                        interrupted = args.interrupt_after_press is not None and control.press_steps>=args.interrupt_after_press
                        if interrupted and closure_step is None:
                            closure_step = step
                        frame = browser.call('render', state={'time': world.time, 'links': world.links(),
                                             'eye_condition': 'closed' if interrupted else condition,'capture_robot_mask':bool(args.rgb_reference_head)}, capture=True)
                        left, right = (base64.b64decode(frame[k]) for k in ('left', 'right'))
                        prediction = None
                        position_valid=True
                        if args.oracle_diagnostic:
                            eye = np.array(to_physics(geometry['eye']))
                            opening = frame['open']
                        else:
                            try:
                                prediction = model.predict(left, right, native_face=base64.b64decode(frame['face']) if args.native_face else None)
                            except ValueError as error:
                                errors.append(str(error))
                                break
                            eye = np.array(to_physics(prediction['eye_scene']))
                            opening = prediction['openness']
                            if reference_guard is not None:
                                images=[np.array(Image.open(BytesIO(blob)).convert('RGB')) for blob in (left,right)]
                                masks=[np.array(Image.open(BytesIO(base64.b64decode(frame['robot_mask_'+view]))).convert('L'))>0 for view in ('left','right')]
                                observed=reference_guard.update(images,masks,opening,step/ACTION_HZ)
                                position_valid=observed['valid'];prediction['observed_reference']=observed
                                if position_valid:
                                    prediction['canonical_framewise_eye_scene']=prediction['eye_scene']
                                    prediction['eye_scene']=observed['point_scene'];eye=np.array(to_physics(observed['point_scene']))
                            if position_guard is not None:
                                guarded=position_guard.update(prediction['eye_scene'],opening,step/ACTION_HZ)
                                position_valid=guarded is not None
                                prediction['temporal_position']={'valid':position_valid,'point_scene':guarded}
                                if position_valid:eye=np.array(to_physics(guarded))
                        target = eye+np.array([0., 0., TARGET_HEIGHT_M])
                        aligned = bool(position_valid and np.linalg.norm(np.array(world.nozzle())-target)<=ALIGN_M)
                        decision = control.decide(openness=opening, gap=world.own_state()[7], aligned=aligned,position_valid=position_valid if reference_guard is not None else True)
                        if not position_valid and decision['arm']=='target':
                            raise ValueError('無効な画像位置への接近指令です')
                        own_joints = world.own_state()[:7]
                        if decision['arm']=='target':
                            held = None
                            joints = world.ik(target)
                            if np.max(np.abs(np.array(own_joints)-path[-1]))>PATH_SPACING_RAD:
                                path.append(own_joints)
                        elif decision['arm']=='hold':
                            if held is None:
                                held = own_joints
                            joints = held
                        else:
                            if len(path)>1 and np.max(np.abs(np.array(own_joints)-path[-1]))<=WAYPOINT_TOLERANCE_RAD:
                                path.pop()
                            joints = path[-1]
                        raw_joints=list(joints)
                        if args.synchronize_arm_targets and decision['arm']=='target':
                            try:
                                joints=synchronized_joint_target(own_joints,joints,args.joint_velocity_cap/ACTION_HZ,
                                    [(world.joint_info[i][8],world.joint_info[i][9]) for i in range(ARM_JOINT_COUNT)])
                            except ValueError as error:
                                errors.append(str(error));break
                        action = [*joints, decision['gap']]
                        observation = recorder.before_action(step, world.own_state(), action) if recorder else None
                        before = world.time
                        try:
                            outcome = world.step(action, audit_pressure=True,audit_motion=args.audit_motion)
                        except ValueError as error:
                            errors.append(str(error))
                            break
                        samples = browser.call('pressureSamples', samples=outcome['pressure_samples'])
                        if moving:
                            domain_samples.extend(outcome['drop_domain_samples'])
                            motion_samples.extend(dict(row,phase=decision['phase']) for row in outcome['motion_samples'])
                        for sample in samples:
                            pressure_samples[round(sample['time']*PHYSICS_HZ)] = {**sample, 'command_gap_m': action[7]}
                        interval = None if outcome['contacts'] else audit_pressure_interval(samples,
                            start_time=before, end_time=before+1/ACTION_HZ, command_gap_m=action[7],
                            press_threshold_m=BOTTLE_DEPTH_M-PRESS_TOLERANCE_M, physics_hz=PHYSICS_HZ,
                            open_threshold=open_threshold)
                        contacts += outcome['contacts']
                        if recorder:
                            demonstration.append(observation)
                            recorder.after_action(step, world.own_state(), action)
                        hits += browser.call('scoreSegments', segments=outcome.get('segments', []))
                        rows.append({'step': step, 'decision': decision, 'prediction': prediction,
                                     'audit_open': frame['open'],
                                     'audit_position_error_m':float(np.linalg.norm(np.array(prediction['eye_scene'])-scoring_eye)) if scoring_eye is not None and prediction is not None else None,
                                     'own': world.own_state(), 'nozzle': world.nozzle(), 'aligned': aligned,
                                     'action': action,'raw_joint_target':raw_joints, 'pressure_interval': interval, 'contacts': outcome['contacts']})
                        if recorder or step%20==0:
                            (directory/f'left-{step:03}.png').write_bytes(left)
                            (directory/f'right-{step:03}.png').write_bytes(right)
                        if contacts:
                            break
                    closed = [s for s in pressure_samples.values() if s['open']<open_threshold]
                    unsafe_command = sum(r['pressure_interval']['closed_command_samples'] for r in rows if r['pressure_interval'] is not None)
                    unsafe_measured = sum(s['gap_m']<BOTTLE_DEPTH_M-PRESS_TOLERANCE_M for s in closed)
                    residual_max = max((max(0., BOTTLE_DEPTH_M-s['gap_m'])*1000 for s in closed), default=0.)
                    retreat_error = float(np.linalg.norm(np.array(world.nozzle())-start_nozzle))
                    physical_domain=domain_summary(domain_samples,expected_intervals=args.frames*PHYSICS_HZ//ACTION_HZ) if moving else None
                    task=fixed_motion_task(condition,len(world.drops),hits,args.drop_model,physical_domain)
                    closed_positive_pressure=0
                    closed_flow_ml=0.
                    pressure_coverage_complete=True
                    if moving:
                        pressure_by_tick={}
                        for record in motion_samples:
                            for time_key,bottle_key in (('start_time','bottle_before'),('time','bottle_after')):
                                pressure_by_tick[round(record[time_key]*PHYSICS_HZ)]=record[bottle_key]['pressure_kpa']
                        pressure_coverage_complete=all(tick in pressure_by_tick for tick in pressure_samples)
                        closed_positive_pressure=(sum(pressure_by_tick[round(s['time']*PHYSICS_HZ)]>PRESSURE_EPSILON_KPA for s in closed)
                                                  if pressure_coverage_complete else None)
                        closed_flow_ml=sum(record['bottle_before']['reservoir_ml']-record['bottle_after']['reservoir_ml']
                            for record in motion_samples if pressure_samples[round(record['time']*PHYSICS_HZ)]['open']<open_threshold)
                    accepted = (len(rows)==args.frames and not contacts and not errors and not control.aborted
                                and not unsafe_command and not unsafe_measured and pressure_coverage_complete
                                and not closed_positive_pressure and closed_flow_ml==0. and task
                                and (not moving or condition!='natural' or control.phase=='retreat')
                                and abs(world.own_state()[7]-rest_gap)<PRESS_TOLERANCE_M and retreat_error<ALIGN_M)
                    result = {'person':args.person,'condition': condition, 'case': case['id'], 'frames': len(rows), 'accepted': bool(accepted),
                              'oracle_diagnostic': args.oracle_diagnostic, 'contacts': contacts, 'drops': len(world.drops),
                              'errors': errors,
                              'hits': hits, 'closed_command_samples': unsafe_command, 'closed_measured_samples': unsafe_measured,
                              'closed_residual_max_mm': residual_max, 'closed_residual_seconds': unsafe_measured/PHYSICS_HZ,
                              'final_gap_m': world.own_state()[7], 'retreat_error_m': retreat_error,
                              'unannounced_closure_step': closure_step,
                              'final_phase': control.phase, 'aborted': control.aborted}
                    result.update(drop_model=args.drop_model,drop_domain=physical_domain,open_threshold=open_threshold,
                                  closed_positive_pressure_samples=closed_positive_pressure if moving else None,
                                  closed_flow_ml=closed_flow_ml if moving else None,
                                  pressure_coverage_complete=pressure_coverage_complete if moving else None,
                                  start_pose=args.start_pose,initial_joints=list(initial),start_pose_audit=start_audit)
                    atomic_json(directory/'trace.json', rows)
                    if recorder:
                        atomic_json(directory/'frames.json', demonstration)
                    atomic_json(directory/'pressure-samples.json', list(pressure_samples.values()))
                    if moving:
                        atomic_json(directory/'motion-samples.json',motion_samples)
                        atomic_json(directory/'drop-domain-samples.json',domain_samples)
                    atomic_json(directory/'result.json', result)
                    results.append(result)
                    atomic_json(output/'result.json', {'completed': False, 'training_updates': 0, 'rows': results})
                    print(json.dumps(result), flush=True)
                finally:
                    world.close()
        atomic_json(output/'result.json', {'completed': True, 'training_updates': 0,
                    'oracle_diagnostic': args.oracle_diagnostic, 'rows': results,
                    'passed': all(r['accepted'] for r in results), 'seconds': time.monotonic()-started})
    finally:
        browser.close()


if __name__=='__main__':
    main()
