"""固定した開発条件を逐次記録し、合格した実演を同期教材へまとめる。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from budget_cycle import experiment_timeout
from resumable_job import atomic_json, run_job, status
from eyedrop.smolvla_observation import command_observation_contract

PEOPLE = ('original', 'seed-san', 'godette')
SPLITS = {'train': ('train-00', 'train-04'), 'validation': ('val-00', 'val-01')}
FRAMES = 1200
PYTHON = '.venv-learning/bin/python'
JOBS = Path('outputs/autonomous-jobs')
PERSON_SECONDS = 900
AUDIT_SECONDS = 120
INSTRUCTION = 'Place one drop in the left eye, release and retreat.'
COMMON = ['--native-face', '--position-mode', 'full-stereo', '--all-position-head',
          'outputs/learning/panda-safe-home-fit-01', '--diagnostic-ready-head',
          'outputs/learning/panda-local-ready-head-01', '--bottle-input', 'measured',
          '--finger-press-speed', '0.0005', '--finger-release-speed', '0.02',
          '--press-steps-limit', '24', '--standby-gap', '0.0169', '--frames', str(FRAMES),
          '--joint-velocity-cap', '0.1', '--continue-approach-after-first-open',
          '--synchronize-arm-targets', '--affine-blink-envelope', '--audit-target-v2',
          '--guard-prepare-contact', '--rgb-reference-head',
          'outputs/learning/panda-visible-reference-fit-01', '--natural-only',
          '--diagnostic-extra-cases', 'train-00', 'train-04', '--drop-model',
          'moving-nozzle-v2', '--audit-motion', '--start-pose', 'visible-downward',
          '--record-demonstration']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy'); parser.add_argument('output')
    args = parser.parse_args()
    policy_path, out = Path(args.policy), Path(args.output)
    active = status(JOBS)
    if active['locked'] or (active['current'] and active['current']['live_descendants']):
        raise RuntimeError('別の実験が実行中です')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(PYTHONPATH='scripts:outputs/runtime/review-sklearn-v1/src',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    original = json.loads(Path('outputs/learning/panda-cartesian-data-02/natural/protocol.json').read_bytes())
    cases = {r['id']: r for split in SPLITS for r in original[split]}
    plan = {split: [dict(cases[case], id=f'{person}-{case}', person=person, source_case=case)
                   for person in PEOPLE for case in selected] for split, selected in SPLITS.items()}
    plan.update(frames=FRAMES, fps=20, rest_gap_m=.0343, language_instruction=INSTRUCTION,
                observation_history_contract=command_observation_contract(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='fresh fixed-controller demonstrations, not learned motion success')
    atomic_json(out/'protocol.json', plan)
    receipts = []
    for person in PEOPLE:
        dest = out/'source'/person
        stages = [(f'sync-teacher-{person}-01', [PYTHON, 'scripts/evaluate_panda_fixed_rgb.py', str(dest), '--person', person, *COMMON], PERSON_SECONDS, dest/'result.json'),
                  (f'sync-audit-{person}-01', [PYTHON, 'scripts/reference_strict95.py', str(out/'audit'/person), str(dest)], AUDIT_SECONDS, out/'audit'/person/'result.json')]
        for job_id, command, seconds, result_path in stages:
            parent = json.loads(policy_path.read_bytes())
            parent['job_ids'].append(job_id); parent['evidence'].append(str(result_path))
            atomic_json(policy_path, parent)
            atomic_json(out/'progress.json', dict(stage=job_id, completed=False, receipts=receipts))
            receipt = run_job(JOBS, job_id, command, experiment_timeout(parent, datetime.now(timezone.utc), seconds))
            receipts.append(receipt)
            if receipt['returncode'] or receipt['live_descendants']:
                raise RuntimeError('実行失敗。終了記録とログを確認してください')
            result = json.loads(result_path.read_bytes())
            if job_id.startswith('sync-teacher') and not (result['completed'] and result['passed'] and len(result['rows']) == 4):
                raise ValueError('固定教師の4条件が点眼成立していません。原因確認へ切り替えます')
    episodes = []
    for split in SPLITS:
        for item in plan[split]:
            source = out/'source'/item['person']/f'natural-{item["source_case"]}'
            saved = json.loads((source/'result.json').read_bytes())
            if not saved['accepted']:
                raise ValueError('不合格実演を教材へ含めません')
            dest = out/item['id']; dest.mkdir()
            frames = json.loads((source/'frames.json').read_bytes())
            if len(frames) != FRAMES:
                raise ValueError('全操作の観測が必要です')
            for frame in frames:
                for side in ('left', 'right'):
                    image = source/frame[side]
                    if not image.is_file():
                        raise ValueError('画像が不足しています')
                    os.link(image, dest/image.name)
                    frame[side] = f'{item["id"]}/{image.name}'
            atomic_json(dest/'frames.json', frames)
            episode = dict(id=item['id'], split=split, accepted=True, frames=FRAMES,
                           original_source=str(source), initial_gap_command_m=plan['rest_gap_m'],
                           source_trace_sha256=hashlib.sha256((source/'trace.json').read_bytes()).hexdigest())
            atomic_json(dest/'result.json', episode); episodes.append(episode)
    atomic_json(out/'result.json', dict(accepted=True, completed=True, episodes=episodes,
                                       language_instruction=INSTRUCTION, learned_policy_success=False))
    atomic_json(out/'progress.json', dict(completed=True, receipts=receipts))


if __name__ == '__main__':
    main()
