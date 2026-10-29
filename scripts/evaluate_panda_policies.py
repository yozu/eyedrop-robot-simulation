"""同じ実験内で固定ルール・追加学習前後を全動作比較する。"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from panda_rpc import PandaBrowser
from resumable_job import atomic_json
from eyedrop.panda_world import PandaWorld,REST_GAP_M,BOTTLE_DEPTH_M,to_physics,PHYSICS_HZ,ACTION_HZ
from eyedrop.smolvla_observation import history_from_normalization
from pressure_interval_audit import audit_pressure_interval
from cartesian_action import action_dimension,CARTESIAN_DIM
from endpoint_policy import MODE as ENDPOINT_MODE

PRESS_TOLERANCE_M=.0001


class PolicyWorker:
    def __init__(self,root,normalization,adapter,log,native_full=False,action_steps=1,noise_mode='fresh',gripper_tree=None,prediction_chunk_size=None,endpoint_regression=False,relative_initial_sha256=None):
        if prediction_chunk_size is not None and (type(prediction_chunk_size) is not int or prediction_chunk_size not in (1,50)):
            raise ValueError('予測動作数は1または50です')
        if relative_initial_sha256 and (not native_full or not adapter or prediction_chunk_size!=1 or action_steps!=1 or gripper_tree or endpoint_regression):
            raise ValueError('相対指令の初期比較は通常重み・1予測1実行を明示してください')
        self.log=open(log,'w')
        command=[str(root/'.venv-lerobot-smolvla/bin/python'),str(root/'scripts/panda_policy_worker.py'),str(normalization),'--action-steps',str(action_steps),'--noise-mode',noise_mode]
        if adapter:command+=['--full-policy' if native_full else '--adapter',str(adapter)]
        if gripper_tree:command+=['--gripper-tree',str(gripper_tree)]
        if prediction_chunk_size is not None:command+=['--prediction-chunk-size',str(prediction_chunk_size)]
        if endpoint_regression:command+=['--endpoint-regression']
        if relative_initial_sha256:command+=['--relative-initial-sha256',relative_initial_sha256]
        environment={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=environment)
        response=self.process.stdout.readline()
        if not response or not json.loads(response).get('ready'):raise RuntimeError('モデルの起動失敗')
        if relative_initial_sha256 and json.loads(response).get('relative_initial_sha256')!=relative_initial_sha256:
            self.close()
            raise RuntimeError('要求した初期重みの対応付けが確認できません')
        if prediction_chunk_size is not None and json.loads(response).get('prediction_chunk_size')!=prediction_chunk_size:
            self.close()
            raise RuntimeError('要求した予測動作数とモデルが一致しません')
        if endpoint_regression and (json.loads(response).get('policy_mode')!=ENDPOINT_MODE or json.loads(response).get('num_steps')!=1 or json.loads(response).get('noise_mode')!='zero'):
            self.close()
            raise RuntimeError('要求した1段回帰の推論方式が一致しません')

    def call(self,**request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        response=self.process.stdout.readline()
        if not response:raise RuntimeError('推論ワーカーが終了しました')
        value=json.loads(response)
        if 'error' in value:raise RuntimeError(value['error'])
        return value

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"operation":"close"}\n');self.process.stdin.flush();self.process.stdin.close()
            self.process.wait(timeout=30)
        self.log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw');parser.add_argument('dataset')
    parser.add_argument('adapter');parser.add_argument('output');parser.add_argument('--native-full',action='store_true')
    parser.add_argument('--action-steps',type=int,choices=(1,2,5),default=1)
    parser.add_argument('--gripper-tree')
    parser.add_argument('--interval-audit',action='store_true')
    parser.add_argument('--cartesian-actions',action='store_true')
    parser.add_argument('--match-prediction-chunks',action='store_true')
    parser.add_argument('--endpoint-regression',action='store_true')
    parser.add_argument('--noise-mode',choices=('fresh','fixed','zero'),default='fresh');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    raw=Path(args.raw).resolve();data=Path(args.dataset).resolve();adapter=Path(args.adapter).resolve()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((raw/'protocol.json').read_text())
    expected_cartesian=action_dimension(json.loads((data/'normalization.json').read_text()))==CARTESIAN_DIM
    if args.cartesian_actions!=expected_cartesian:raise ValueError('動作空間の明示指定と統計が異なります')
    if args.cartesian_actions and not args.interval_audit:raise ValueError('Cartesian比較は全物理区間の押圧監査が必要です')
    if args.match_prediction_chunks and not args.native_full:raise ValueError('今回の予測数一致比較は通常モデル専用です')
    if args.endpoint_regression!=(args.noise_mode=='zero') or (args.endpoint_regression and (not args.native_full or not args.cartesian_actions or not args.match_prediction_chunks or args.action_steps!=1)):
        raise ValueError('1段回帰はCartesian通常モデル/先頭1動作/chunk一致/zeroを明示してください')
    matching_chunk=json.loads((adapter/'config.json').read_text())['chunk_size'] if args.match_prediction_chunks else None
    if args.gripper_tree and not args.native_full:raise ValueError('分類木は対応する通常モデルと組み合わせてください')
    rest_gap=protocol.get('rest_gap_m',REST_GAP_M)
    weights_file=adapter/('model.safetensors' if args.native_full else 'adapter_model.safetensors')
    with weights_file.open('rb') as stream:weights_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    initial=json.loads((raw/'train-02/result.json').read_text())['initial_joints']
    fixed=json.loads((raw/'train-02/frames.json').read_text())
    atomic_json(output/'protocol.json',{'cases':protocol['validation'],'policies':['fixed-rule','pretrained','finetuned'],
                'action_space':'nozzle XYZ and gap to robot-only IK' if args.cartesian_actions else 'joint angles and gap',
                'prediction_chunks':{'fixed-rule':None,'pretrained':matching_chunk if matching_chunk is not None else 50,'finetuned':json.loads((adapter/'config.json').read_text())['chunk_size']} if args.native_full else None,
                'matching_prediction_chunks':args.match_prediction_chunks,
                'policy_algorithm':ENDPOINT_MODE if args.endpoint_regression else 'smolvla-flow-matching-v1',
                'pretrained_comparator':'同じ1段/zeroへ適応した追加学習前' if args.endpoint_regression else '標準flowの追加学習前',
                'task_policy_sha256':hashlib.sha256((adapter/'task-policy.json').read_bytes()).hexdigest() if args.endpoint_regression else None,
                'action_steps':args.action_steps,'queue':'1は従来の先頭再推論。2/5はSmolVLAPolicy.select_actionの公式キュー。乱数は物理時点ごとに進める',
                'noise_mode':args.noise_mode,'noise_seed':'714+case_index。fixedは試行の最初のnoiseを再利用し、都合の良いseedを選別しない',
                'interval_pressure_audit_hz':PHYSICS_HZ if args.interval_audit else None,
                'press_threshold_m':BOTTLE_DEPTH_M-PRESS_TOLERANCE_M,'safety':'0.1mm以上の容器壁押圧指令と実測押圧を別計数。接触余裕は教師と同じ',
                'checkpoint_format':'native-full-float32' if args.native_full else 'peft-lora','weights_sha256':weights_hash,
                'adapter_sha256':None if args.native_full else weights_hash,
                'gripper_tree':{'path':str(Path(args.gripper_tree).resolve()),'sha256':hashlib.sha256(Path(args.gripper_tree).read_bytes()).hexdigest(),
                                'scope':'finetunedだけをSmolVLA関節出力と学習分類木の指幅へ変更。診断目的・未採用'} if args.gripper_tree else None,
                'normalization_sha256':hashlib.sha256((data/'normalization.json').read_bytes()).hexdigest(),
                'raw_protocol_sha256':hashlib.sha256((raw/'protocol.json').read_bytes()).hexdigest(),
                'source_sha256':{relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in
                    ('scripts/evaluate_panda_policies.py','scripts/panda_policy_worker.py','scripts/smolvla_runtime.py',
                     'scripts/pressure_interval_audit.py','scripts/panda_browser.cjs',
                     'scripts/cartesian_action.py','scripts/action_encoding.py',
                     'scripts/endpoint_policy.py',
                     'scripts/tree_gripper.py','scripts/direct_gripper.py',
                     'src/eyedrop/panda_world.py','src/eyedrop/smolvla_observation.py','preview/panda-runtime.js','preview/scene.js')},
                'fixed_rule':'学習用のnominal train-02の300関節・指幅指令を固定再生。検証の正解は制御へ渡さない'})
    browser=PandaBrowser(output/'browser.log');results=[];started=time.monotonic()
    try:
        setup=PandaWorld(output,rest_gap=rest_gap);browser.call('setup',shapes=setup.drawing());setup.close()
        for mode in ('fixed-rule','pretrained','finetuned'):
            worker=None
            try:
                if mode!='fixed-rule':worker=PolicyWorker(root,data/'normalization.json',adapter if mode=='finetuned' else None,output/f'{mode}.log',native_full=args.native_full and mode=='finetuned',action_steps=args.action_steps,noise_mode=args.noise_mode,gripper_tree=args.gripper_tree if mode=='finetuned' else None,prediction_chunk_size=matching_chunk,endpoint_regression=args.endpoint_regression)
                for case_index,case in enumerate(protocol['validation']):
                    directory=output/f'{mode}-{case["id"]}';directory.mkdir()
                    browser.call('reset',depth=case['depth_m'],phase=case['blink_phase_s'])
                    geometry=browser.call('geometry')
                    world=PandaWorld(directory,rest_gap=rest_gap)
                    try:
                        world.reset(initial);world.add_geometry(geometry)
                        history=history_from_normalization(json.loads((data/'normalization.json').read_bytes()),rest_gap)
                        history.reset(world.own_state(),rest_gap)
                        if worker:worker.call(operation='reset',seed=714+case_index)
                        rows=[];hits=[];contacts=[];closed_command=0;closed_measured=0;error=None;interval_failures=0
                        for step in range(300):
                            frame=browser.call('render',state={'time':world.time,'links':world.links()},capture=True)
                            state=history.observation()
                            for side in ('left','right'):(directory/f'{side}-{step:03}.png').write_bytes(base64.b64decode(frame[side]))
                            inference_started=time.monotonic()
                            action=None
                            try:
                                action=worker.call(operation='act',state=state,left=frame['left'],right=frame['right'])['action'] if worker else fixed[step]['action']
                                predicted_cartesian=None
                                if args.cartesian_actions and worker:
                                    predicted_cartesian=action
                                    action=[*world.ik(predicted_cartesian[:3]),predicted_cartesian[-1]]
                                inference_s=time.monotonic()-inference_started
                                if frame['open']<.9:
                                    closed_command+=int(action[7]<BOTTLE_DEPTH_M-PRESS_TOLERANCE_M)
                                    closed_measured+=int(world.own_state()[7]<BOTTLE_DEPTH_M-PRESS_TOLERANCE_M)
                                interval_start=world.time
                                outcome=world.step(action,audit_pressure=True) if args.interval_audit else world.step(action)
                                interval_result=None;pressure_samples=None
                                if args.interval_audit:
                                    pressure_samples=browser.call('pressureSamples',samples=outcome['pressure_samples'])
                                    if outcome['contacts']:
                                        interval_result={'complete':False,'passed':False,'reason':'safety-stop'}
                                    else:
                                        interval_result=audit_pressure_interval(pressure_samples,start_time=interval_start,
                                            end_time=interval_start+1/ACTION_HZ,command_gap_m=action[7],
                                            press_threshold_m=BOTTLE_DEPTH_M-PRESS_TOLERANCE_M,physics_hz=PHYSICS_HZ)
                                        interval_result['complete']=True
                                    interval_failures+=int(not interval_result['passed'])
                                history.append(step+1,world.own_state(),action[7])
                                hits+=browser.call('scoreSegments',segments=outcome.get('segments',[]))
                                contacts+=outcome['contacts']
                                row={'step':step,'state':state,'action':action,'open_audit':frame['open'],
                                     'inference_seconds':inference_s,'contacts':outcome['contacts']}
                                if args.interval_audit:row.update(pressure_interval_audit=interval_result,pressure_samples=pressure_samples)
                                if predicted_cartesian is not None:row['predicted_cartesian_action']=predicted_cartesian
                                rows.append(row)
                                if contacts:break
                            except (ValueError,RuntimeError) as failure:
                                error=repr(failure)
                                rows.append({'step':step,'state':state,'action':action,'open_audit':frame['open'],
                                             'failed':True,'error':error,'own_after_failure':world.own_state()})
                                break
                        nominal=json.loads((raw/'train-02/scoring-geometry.json').read_text())['eye']
                        retreat=np.array(to_physics(nominal))+np.array([0,0,protocol['initial_nozzle_height_m']])
                        retreat_error=float(np.linalg.norm(np.array(world.nozzle())-retreat))
                        gap=world.own_state()[7]
                        accepted=len(rows)==300 and error is None and not contacts and not closed_command and not closed_measured and not interval_failures and len(world.drops)==1 and len(hits)==1 and hits[0]['result']=='target' and hits[0]['open']>=.9 and abs(gap-rest_gap)<.0001 and retreat_error<.001
                        result={'mode':mode,'case':case,'accepted':accepted,'frames':len(rows),'error':error,
                                'contacts':contacts,'closed_press_commands':closed_command,'closed_measured_compression':closed_measured,
                                'drops':len(world.drops),'hits':hits,'released_gap':gap,'retreat_error_m':retreat_error}
                        if args.interval_audit:result.update(interval_pressure_audit_hz=PHYSICS_HZ,failed_pressure_intervals=interval_failures)
                        atomic_json(directory/'trace.json',rows);atomic_json(directory/'result.json',result)
                        results.append(result);atomic_json(output/'result.json',{'rows':results,'seconds':time.monotonic()-started,'evaluation_completed':len(results)==6,'adopted':False})
                        browser.call('screenshot',path=str(directory/'final.png'))
                        print(json.dumps(result),flush=True)
                    finally:world.close()
            finally:
                if worker:worker.close()
    finally:browser.close()


if __name__=='__main__':main()
