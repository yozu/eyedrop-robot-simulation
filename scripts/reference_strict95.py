"""既存の0.95開眼基準を、位置参照候補にも同じ形で適用する。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from resumable_job import atomic_json

STRICT_OPEN = .95
COMPRESSED_GAP_M = .0158
LIMIT_MM = 1.


def score_strict95(task, complete, bottle_rows, samples):
    bottles={(r['condition'],r['case']):r for r in bottle_rows};rows=[]
    for item in task:
        key=item['condition'],item['case'];closed=[s for s in samples[key] if s['open']<STRICT_OPEN]
        commanded=sum(s['command_gap_m']<COMPRESSED_GAP_M for s in closed)
        measured=sum(s['gap_m']<COMPRESSED_GAP_M for s in closed)
        landing=item['condition']!='natural' or len(item['hits'])==1 and item['hits'][0]['open']>=STRICT_OPEN
        bottle=bottles[key]
        fluid_zero=bottle['drop_count_matches'] and bottle['closed_positive_pressure_seconds']==0 and bottle['closed_flow_ml']==0
        rows.append({'condition':key[0],'case':key[1],'full_trace':complete[key],'closed_command95':commanded,'closed_measured95':measured,
            'open_landing95':landing,'closed_fluid_pressure_zero95':bool(fluid_zero),'closed_hits95':sum(h['open']<STRICT_OPEN for h in item['hits']),
            'accepted':bool(item['accepted'] and complete[key] and not commanded and not measured and landing and fluid_zero)})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('trials',nargs='+');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    atomic_json(out/'protocol.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'open_threshold':STRICT_OPEN,
        'compressed_gap_m':COMPRESSED_GAP_M,'training_updates':0,'trials':args.trials,'criteria':'original strict95;no boundary exclusion or pressure reset'})
    (out/'source.py').write_bytes(Path(__file__).read_bytes());results={}
    for name in args.trials:
        trial=Path(name);task=json.loads((trial/'result.json').read_bytes())
        if not task['completed']:raise ValueError('全動作試験が未完了です')
        if not (trial/'pressure-audit.json').exists():
            subprocess.run([sys.executable,str(root/'scripts/audit_panda_fixed_pressure.py'),str(trial)],cwd=root,check=True)
        pressure=json.loads((trial/'pressure-audit.json').read_bytes())
        if not pressure['records_verified']:raise ValueError('全物理時点が不一致です')
        bottle_file=trial/'bottle-pressure-audit-095.json'
        if bottle_file.exists():raise ValueError('0.95監査が既に存在します。重複実行せず照合してください')
        subprocess.run([sys.executable,str(root/'scripts/audit_panda_bottle_pressure.py'),str(trial),'--input-contract','measured','--open-threshold',str(STRICT_OPEN)],cwd=root,check=True)
        bottle=json.loads(bottle_file.read_bytes())
        if not bottle['replay_drop_counts_match']:raise ValueError('容器モデルの再生が不一致です')
        complete={(r['condition'],r['case']):r['complete'] and r['matches_evaluator'] for r in pressure['rows']}
        samples={};positions={};hashes={}
        for item in task['rows']:
            key=item['condition'],item['case'];folder=trial/f'{key[0]}-{key[1]}'
            physical=folder/'pressure-samples.json';trace=folder/'trace.json'
            samples[key]=json.loads(physical.read_bytes());frames=json.loads(trace.read_bytes())
            for p in (physical,trace):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            used=[r for r in frames if r.get('prediction') and r['prediction'].get('observed_reference',{}).get('valid')]
            positions['/'.join(key)]={'count':len(used),'max_mm':max(r['audit_position_error_m'] for r in used)*1000 if used else None}
        rows=score_strict95(task['rows'],complete,bottle['rows'],samples)
        natural_positions=[positions[r['condition']+'/'+r['case']] for r in rows if r['condition']=='natural']
        results[str(trial)]={'rows':rows,'accepted':sum(r['accepted'] for r in rows),'count':len(rows),'positions':positions,
            'position_gate_passed':all(x['max_mm'] is not None and x['max_mm']<=LIMIT_MM for x in natural_positions) if natural_positions else None,
            'bottle':bottle,'input_sha256':hashes,'task_sha256':hashlib.sha256((trial/'result.json').read_bytes()).hexdigest()}
    atomic_json(out/'result.json',{'completed':True,'trials':results,'adopted':False,'training_updates':0})
    print(json.dumps({k:{'accepted95':v['accepted'],'count':v['count'],'position_gate':v['position_gate_passed']} for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
