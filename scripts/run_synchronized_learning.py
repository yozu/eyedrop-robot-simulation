"""同期教材の品質確認から一候補の学習・固定比較まで逐次進める。"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from budget_cycle import experiment_timeout
from resumable_job import atomic_json, run_job, status, process_info

JOBS=Path('outputs/autonomous-jobs')
PYTHON='.venv-lerobot-smolvla/bin/python'
RAW=Path('outputs/learning/panda-synchronized-data-01')
DATA=Path('outputs/learning/panda-synchronized-dataset-01')
BASE=Path('outputs/diagnostics/smolvla-synchronized-connection-01')
PAIR=Path('outputs/diagnostics/smolvla-synchronized-pair-01')
TRAIN=Path('outputs/learning/smolvla-synchronized-01')
CHECK=Path('outputs/diagnostics/smolvla-synchronized-check-01')
MOTION=Path('outputs/diagnostics/smolvla-synchronized-motion-01')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('policy');parser.add_argument('output')
    parser.add_argument('--wait-for-collector',type=Path)
    parser.add_argument('--connection-output',type=Path,default=BASE)
    parser.add_argument('--connection-job',default='sync-connection-01')
    parser.add_argument('--reuse-baseline',type=Path)
    args=parser.parse_args();policy_path=Path(args.policy);out=Path(args.output)
    base=args.connection_output
    out.mkdir(parents=True,exist_ok=True)
    with (out/'cycle.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.wait_for_collector:
            launch=json.loads(args.wait_for_collector.read_bytes())
            atomic_json(out/'waiting.json',dict(pid=os.getpid(),collector=launch,command=sys.argv))
            while not (RAW/'result.json').exists():
                experiment_timeout(json.loads(policy_path.read_bytes()),datetime.now(timezone.utc),15)
                if process_info(launch['pid'])!=launch['process_identity']:
                    raise RuntimeError('教材収集プロセスが終了または変化しました。結果未完了のまま次へ進みません')
                time.sleep(15)
        active=status(JOBS)
        if active['locked'] or (active['current'] and active['current']['live_descendants']):
            raise RuntimeError('別の重い処理が実行中です')
        collected=json.loads((RAW/'result.json').read_bytes())
        if not collected['completed'] or not collected['accepted']:
            raise ValueError('教材の記録が未完了です')
        os.environ.update(PYTHONPATH='scripts:outputs/runtime/review-sklearn-v1/src',
                          PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        stages=[
            dict(id='sync-export-01',output=str(DATA),seconds=1200,command=[PYTHON,'scripts/export_panda_dataset.py',str(RAW),str(DATA)]),
            dict(id=args.connection_job,output=str(base),seconds=1200,command=[PYTHON,'scripts/prepare_synchronized_training.py',str(RAW),str(DATA),args.policy,str(base)]+(['--reuse-baseline',str(args.reuse_baseline)] if args.reuse_baseline else [])),
            dict(id='sync-pair-connection-01',output=str(PAIR),seconds=600,command=[PYTHON,'scripts/prepare_paired_switch_training.py',str(base/'pair-proposal.json'),args.policy,str(base/'pairs.json'),str(PAIR),'--paired-switch-objective','paired-flow-grid']),
            dict(id='sync-training-01',output=str(TRAIN),seconds=3900,command=[PYTHON,'scripts/train_smolvla_relative_sequence.py',str(PAIR),'relative',str(TRAIN),'--paired-switch-manifest',str(base/'pairs.json'),'--paired-switch-objective','paired-flow-grid']),
            dict(id='sync-check-01',output=str(CHECK),seconds=1500,command=[PYTHON,'scripts/evaluate_smolvla_relative_sequence.py',str(TRAIN),str(CHECK)]),
            dict(id='sync-motion-01',output=str(MOTION),seconds=1800,command=[PYTHON,'scripts/evaluate_recovered_smolvla.py',str(RAW),str(DATA),str(TRAIN/'policy-final'),str(MOTION),'--native-full','--initial-policy','outputs/diagnostics/smolvla-official-initial-01/policy','--training-protocol',str(TRAIN/'protocol.json'),'--normalization',str(TRAIN/'policy-final/normalization.json'),'--runtime-src','outputs/runtime/review-sklearn-v1/src'])]
        atomic_json(out/'plan.json',stages)
        atomic_json(out/'runner.json',dict(pid=os.getpid(),command=sys.argv,started_at=datetime.now(timezone.utc).isoformat()))
        decisions=[]
        try:
            for stage in stages:
                result_path=Path(stage['output'])/'result.json';receipt_path=JOBS/stage['id']/'exit.json'
                if receipt_path.exists():
                    receipt=json.loads(receipt_path.read_bytes())
                else:
                    if (JOBS/stage['id']).exists() or Path(stage['output']).exists():
                        raise RuntimeError('中断した出力を自動上書きしません')
                    parent=json.loads(policy_path.read_bytes())
                    timeout=experiment_timeout(parent,datetime.now(timezone.utc),stage['seconds'])
                    if stage['id'] not in parent['job_ids']:parent['job_ids'].append(stage['id'])
                    if str(result_path) not in parent['evidence']:parent['evidence'].append(str(result_path))
                    atomic_json(policy_path,parent)
                    atomic_json(out/'progress.json',dict(completed=False,current=stage['id'],decisions=decisions))
                    receipt=run_job(JOBS,stage['id'],stage['command'],timeout)
                if receipt['returncode'] or receipt['live_descendants'] or not result_path.exists():
                    raise RuntimeError('工程が正常終了していません: '+stage['id'])
                result=json.loads(result_path.read_bytes())
                if stage['id']=='sync-export-01' and not result['accepted']:raise ValueError('教材変換不合格')
                if 'connection' in stage['id'] and not result['ready_for_training']:raise ValueError('モデル接続不合格')
                if stage['id']=='sync-training-01' and not result['completed']:raise ValueError('学習結果未完了')
                decision=dict(id=stage['id'],exit=receipt,result=str(result_path),
                              passed=result.get('robust_BC_passed',result.get('train_BC_passed',result.get('accepted'))),
                              note='精度不合格時の全動作は限定シミュレータ診断。採用判定は変えない。')
                decisions.append(decision)
                with (out/'events.jsonl').open('a',encoding='utf8') as f:
                    f.write(json.dumps(decision,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            atomic_json(out/'result.json',dict(completed=True,decisions=decisions,adopted=False,agent_review_required=True))
            atomic_json(out/'progress.json',dict(completed=True,decisions=decisions,review_due=True))
        except Exception as error:
            atomic_json(out/'failure.json',dict(type=type(error).__name__,message=str(error),decisions=decisions))
            raise


if __name__=='__main__':main()
