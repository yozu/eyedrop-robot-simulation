"""回収実演と同じ機構で、追加学習前後のSmolVLAを全動作比較する。"""
import argparse
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_panda_policies import PolicyWorker
from panda_rpc import PandaBrowser
from resumable_job import atomic_json
from eyedrop.panda_world import PandaWorld, ACTION_HZ, PHYSICS_HZ, BOTTLE_DEPTH_M, SUBSTEPS
from eyedrop.smolvla_observation import history_from_normalization
from eyedrop.bottle import Bottle
from eyedrop.drop_kinematics import MOVING_NOZZLE, DOMAIN, domain_summary
from learned_rollout_fluid import audit_fluid
from relative_joint_action import (MODE as RELATIVE, ACTION_DIM, file_hash, document_hash, verify_initial_policy,
                                   load_policy_binding, validate_policy_binding)

CASES=('original-val-00','seed-san-val-00')
FRAMES=1200
OPEN_THRESHOLD=.95
GAP_TOLERANCE_M=.0001
RETREAT_TOLERANCE_M=.001
PRESSURE_EPSILON_KPA=1e-9
CAPTURE_STRIDE=20
SEED=714
SCORING_VERSION='first-surface-specified-eye-v3'
EXPECTED_EYE='left'


def comparison_spec(normalization,checkpoint,*,native_full=False,initial_policy=None,training_protocol=None):
    """比較の重み・正規化を、推論や物理の起動前に照合する。"""
    normalization=Path(normalization);checkpoint=Path(checkpoint)
    document=json.loads(normalization.read_bytes())
    relative=document.get('joint_action_encoding')==RELATIVE
    initial_sha=None;record_sha=None
    if relative:
        if not native_full or initial_policy is None or training_protocol is None:
            raise ValueError('相対指令の比較には通常重み、学習時初期重み、学習記録が必要です')
        record=json.loads(Path(training_protocol).read_bytes());record_sha=file_hash(training_protocol)
        initial_sha=record['initial_model_sha256']
        if record['normalization_sha256']!=file_hash(normalization):
            raise ValueError('学習時と比較時の正規化が異なります')
        verify_initial_policy(initial_policy,document,initial_sha)
        config=json.loads((checkpoint/'config.json').read_bytes())
        if config.get('chunk_size')!=1 or config.get('output_features',{}).get('action',{}).get('shape')!=[ACTION_DIM]:
            raise ValueError('学習済み重みは1予測・8動作成分である必要があります')
        binding=SimpleNamespace(config=SimpleNamespace(chunk_size=1,n_action_steps=1))
        load_policy_binding(binding,checkpoint);validate_policy_binding(binding,document)
    elif initial_policy is not None or training_protocol is not None:
        raise ValueError('初期重みの対応付けは相対指令の比較だけで指定してください')
    weights=checkpoint/('model.safetensors' if native_full else 'adapter_model.safetensors')
    return dict(baseline='same-training-initial-relative-interpretation' if relative else 'legacy-pretrained',
        normalization_sha256=file_hash(normalization),normalization_document_sha256=document_hash(document),
        training_protocol_sha256=record_sha,finetuned_model_sha256=file_hash(weights),
        workers=[dict(mode='pretrained',checkpoint=str(initial_policy) if relative else None,
                      native_full=relative,relative_initial_sha256=initial_sha),
                 dict(mode='finetuned',checkpoint=str(checkpoint),native_full=native_full,relative_initial_sha256=None)])


def specified_eye_task(drop_count,hits):
    if drop_count!=1 or len(hits)!=1:return False
    hit=hits[0]
    return (hit.get('result')=='target' and hit.get('open',0)>=OPEN_THRESHOLD and
        hit.get('scoring_version')==SCORING_VERSION and
        hit.get('expected_eye')==EXPECTED_EYE and hit.get('hit_eye')==EXPECTED_EYE)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw');parser.add_argument('data');parser.add_argument('adapter');parser.add_argument('output')
    parser.add_argument('--native-full',action='store_true');parser.add_argument('--initial-policy')
    parser.add_argument('--training-protocol');parser.add_argument('--normalization')
    parser.add_argument('--runtime-src',default='outputs/runtime/panda-hybrid-v1/src')
    args=parser.parse_args()
    raw=Path(args.raw);data=Path(args.data);out=Path(args.output)
    normalization=Path(args.normalization) if args.normalization else data/'normalization.json'
    comparison=comparison_spec(normalization,Path(args.adapter),native_full=args.native_full,
        initial_policy=args.initial_policy,training_protocol=args.training_protocol)
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    # 推論子プロセスでは専用環境の依存を使い、親の学習環境を混入させない。
    os.environ['PYTHONPATH']=os.pathsep.join((str(root/'scripts'),str(root/args.runtime_src)))
    plan=json.loads((raw/'protocol.json').read_bytes())
    cases={r['id']:r for r in plan['validation']}
    atomic_json(out/'protocol.json',dict(cases=CASES,frames=FRAMES,open_threshold=OPEN_THRESHOLD,
        modes=['pretrained','finetuned'],control='SmolVLA画像2枚・自己状態32・保存共通指示から関節7/指幅を直接実行。固定押圧ゲートなし',
        comparison=comparison,scoring_version=SCORING_VERSION,expected_eye=EXPECTED_EYE,
        drop_model=MOVING_NOZZLE,drop_domain=DOMAIN,
        reference='raw内の教師を原記録として参照。ケース・初期姿勢・物理設定・採点版の一致は別途照合し、旧fixed24は同条件比較に混ぜない',
        inference_exclusions=['eye_truth','person_id','future','bottle_pressure','drops'],adopted=False))
    results=[]
    for spec in comparison['workers']:
        mode=spec['mode']
        worker=PolicyWorker(root,normalization,Path(spec['checkpoint']) if spec['checkpoint'] else None,
            out/(mode+'.log'),prediction_chunk_size=1,native_full=spec['native_full'],
            relative_initial_sha256=spec['relative_initial_sha256'])
        try:
            for index,ident in enumerate(CASES):
                case=cases[ident];source=json.loads((raw/ident/'result.json').read_bytes())
                meta=json.loads((Path(source['original_source']).parent/'protocol.json').read_bytes())
                directory=out/(mode+'-'+ident);directory.mkdir()
                world=PandaWorld(directory,rest_gap=meta['rest_gap'],bottle_input=meta['bottle_input'],
                    finger_release_speed_m_s=meta['finger_release_speed_m_s'],finger_press_speed_m_s=meta['finger_press_speed_m_s'],
                    joint_velocity_cap=meta['joint_velocity_cap'],drop_model=MOVING_NOZZLE)
                browser=PandaBrowser(directory/'browser.log',person=case['person'],hybrid_rgb=True)
                rows=[];samples=[];hits=[];contacts=[];domain_samples=[];motion_samples=[];error=None;closed_commands=0;closed_pressure=0
                bottle=Bottle()
                try:
                    browser.call('setup',shapes=world.drawing());reset=browser.call('reset',depth=case['depth_m'],phase=case['blink_phase_s'])
                    if reset.get('scoring',{}).get('version')!=SCORING_VERSION or reset['scoring'].get('expected_eye')!=EXPECTED_EYE:
                        raise ValueError('指定眼を判定する採点バージョンが不一致です')
                    geometry=browser.call('geometry');world.reset(meta['initial_joints']);world.add_affine_geometry(geometry,browser.call('blinkGeometry'))
                    start_nozzle=np.asarray(world.nozzle())
                    history=history_from_normalization(json.loads(normalization.read_bytes()),meta['rest_gap'])
                    history.reset(world.own_state(),meta['rest_gap'])
                    worker.call(operation='reset',seed=SEED+index)
                    for step in range(FRAMES):
                        frame=browser.call('render',state=dict(time=world.time,links=world.links(),eye_condition='natural'),capture=True)
                        if step%CAPTURE_STRIDE==0:
                            for side in ('left','right'):(directory/f'{side}-{step:03}.png').write_bytes(base64.b64decode(frame[side]))
                        state=history.observation();action=None
                        try:
                            action=worker.call(operation='act',state=state,left=frame['left'],right=frame['right'])['action']
                            outcome=world.step(action,audit_pressure=True,audit_motion=True)
                            motion_samples.extend(outcome['motion_samples'])
                            domain_samples.extend(outcome['drop_domain_samples'])
                            points=browser.call('pressureSamples',samples=outcome['pressure_samples'])
                            for j,point in enumerate(points):
                                if j:
                                    displacement=max(0.,min((BOTTLE_DEPTH_M-point['gap_m'])*1000,bottle.config.max_displacement_mm))
                                    bottle.step_measured(displacement,1/PHYSICS_HZ)
                                record=dict(point,command_gap_m=action[7],audit_pressure_kpa=bottle.pressure_kpa)
                                samples.append(record)
                                if point['open']<OPEN_THRESHOLD:
                                    closed_commands+=int(action[7]<BOTTLE_DEPTH_M-GAP_TOLERANCE_M)
                                    closed_pressure+=int(bottle.pressure_kpa>PRESSURE_EPSILON_KPA)
                            history.append(step+1,world.own_state(),action[7])
                            contacts+=outcome['contacts'];hits+=browser.call('scoreSegments',segments=outcome.get('segments',[]))
                            rows.append(dict(step=step,state=state,action=action,audit_open=frame['open'],own=world.own_state(),contacts=outcome['contacts']))
                            if contacts:break
                        except (RuntimeError,ValueError) as failure:
                            error=repr(failure);rows.append(dict(step=step,error=error,state=state,action=action));break
                    complete=len(rows)==FRAMES and not error and not contacts
                    gap=world.own_state()[7];retreat=float(np.linalg.norm(np.asarray(world.nozzle())-start_nozzle))
                    task=specified_eye_task(len(world.drops),hits)
                    replay_match=bottle.drop_count==len(world.drops)
                    physical_domain=domain_summary(domain_samples,expected_intervals=FRAMES*SUBSTEPS)
                    fluid=audit_fluid(samples,motion_samples,expected_intervals=FRAMES*SUBSTEPS)
                    accepted=bool(complete and task and not closed_commands and not closed_pressure and replay_match
                        and physical_domain['valid'] and fluid['passed'] and abs(gap-meta['rest_gap'])<GAP_TOLERANCE_M and retreat<RETREAT_TOLERANCE_M)
                    result=dict(mode=mode,case=ident,accepted=accepted,frames=len(rows),complete=bool(complete),
                        error=error,contacts=contacts,drops=len(world.drops),hits=hits,closed_command_samples=closed_commands,
                        closed_positive_pressure_samples=closed_pressure,bottle_drop_replay_matches=replay_match,
                        final_gap_m=gap,retreat_error_m=retreat,scoring_version=SCORING_VERSION,expected_eye=EXPECTED_EYE,
                        drop_model=MOVING_NOZZLE,drop_domain=physical_domain,fluid_audit=fluid)
                    atomic_json(directory/'motion-samples.json',motion_samples)
                    atomic_json(directory/'trace.json',rows);atomic_json(directory/'pressure-samples.json',samples);atomic_json(directory/'result.json',result)
                    atomic_json(directory/'drop-domain-samples.json',domain_samples)
                    results.append(result);atomic_json(out/'progress.json',dict(completed=False,rows=results))
                    print(json.dumps(result),flush=True)
                finally:
                    browser.close();world.close()
        finally:
            worker.close()
    atomic_json(out/'result.json',dict(completed=True,rows=results,adopted=False,
        fixed24_preserved_as_reference=True,fixed24_reference_comparable=False,unseen_evaluated=False,
        scoring_version=SCORING_VERSION,expected_eye=EXPECTED_EYE,comparison=comparison,drop_model=MOVING_NOZZLE,drop_domain=DOMAIN,
        finetuned_pilot_passed=all(r['accepted'] for r in results if r['mode']=='finetuned')))


if __name__=='__main__':main()
