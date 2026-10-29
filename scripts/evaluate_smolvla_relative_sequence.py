"""重みを固定した後に同じ5乱数・全教材で検査する。"""
import argparse
import json
from pathlib import Path
import time

import torch
from relative_joint_action import file_hash
from relative_sequence_training import SEEDS, SPLITS, evaluate
from resumable_job import atomic_json
from smolvla_runtime import load_policy, ModelInput

RELOAD_TOLERANCE = 1e-6


def main():
    parser=argparse.ArgumentParser();parser.add_argument('candidate');parser.add_argument('output');args=parser.parse_args()
    candidate,out=Path(args.candidate),Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    trained=json.loads((candidate/'result.json').read_bytes());protocol=json.loads((candidate/'protocol.json').read_bytes())
    if not trained['completed'] or not trained['weights_fixed_at']:raise ValueError('重みの固定が未完了です')
    connection=Path(protocol['connection']);final=candidate/'policy-final'
    for p,sha in ((final/'model.safetensors',trained['model_sha256']),
                  (connection/'samples.pt',protocol['samples_sha256']),
                  (connection/'labels.json',protocol['labels_sha256'])):
        if file_hash(p)!=sha:raise ValueError('固定済みの入力または重みが変更されています')
    cache=torch.load(connection/'samples.pt',map_location='cpu',weights_only=True)
    labels=json.loads((connection/'labels.json').read_bytes())
    policy=load_policy(full_policy=final,chunk_size=1);inputs=ModelInput(policy,final/'normalization.json')
    summaries={};reload_max=0.
    for seed in SEEDS:
        summaries[str(seed)]={}
        noise=torch.randn(1,1,32,generator=torch.Generator().manual_seed(seed)).to('mps')
        for split in SPLITS:
            result=evaluate(policy,inputs,cache,labels,split,noise)
            atomic_json(out/f'noise-{seed}-{split}.json',result)
            summaries[str(seed)][split]={k:v['metrics'] for k,v in result.items()}
            if seed==SEEDS[0] and split=='train':
                saved=json.loads((candidate/'after-train.json').read_bytes())
                for source in result:
                    for a,b in zip(result[source]['rows'],saved[source]['rows'],strict=True):
                        reload_max=max(reload_max,max(abs(x-y) for x,y in zip(a['prediction'],b['prediction'],strict=True)))
                if reload_max>RELOAD_TOLERANCE:raise ValueError('再読込後の予測が一致しません')
            print(json.dumps({'seed':seed,'split':split,'micro':result['micro']['metrics'],
                'sequence_passed':result['sequence']['metrics']['passed']}),flush=True)
            atomic_json(out/'progress.json',dict(completed=False,seed=seed,split=split))
    if file_hash(final/'model.safetensors')!=trained['model_sha256']:raise ValueError('評価中に重みが変わりました')
    all_passed=all(m['passed'] for splits in summaries.values() for sources in splits.values() for m in sources.values())
    atomic_json(out/'result.json',dict(completed=True,scale=trained['scale'],candidate=str(candidate),
        model_sha256=trained['model_sha256'],frozen_before_validation=trained['weights_fixed_at'],
        summaries=summaries,reload_max_abs_difference=reload_max,seconds=time.monotonic()-start,
        robust_BC_passed=all_passed,fixed_updates_completed=trained['fixed_updates_completed'],
        source_sha256=file_hash(__file__),adopted=False,full_task_success=False,updates=0))
    atomic_json(out/'progress.json',dict(completed=True,result='result.json'))


if __name__=='__main__':main()
