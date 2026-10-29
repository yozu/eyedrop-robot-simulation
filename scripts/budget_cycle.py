"""有限計算枠を実験から振り返りへ切り替える。深い考察はエージェントが行う。"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from resumable_job import atomic_json, run_job

POLL_SECONDS = 30
JOBS_ROOT = Path('outputs/autonomous-jobs')


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('期限にはタイムゾーンが必要です')
    return result


def phase_at(policy, now):
    start, compute, review = (timestamp(policy[k]) for k in
                              ('started_at', 'compute_until', 'review_until'))
    if not start < compute < review or now < start:
        raise ValueError('計算枠の時刻が不正です')
    if now >= review:
        return 'review_overdue'
    return 'review_due' if now >= compute else 'compute'


def experiment_timeout(policy, now, requested):
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('有限の実行上限が必要です')
    if phase_at(policy, now) != 'compute':
        raise ValueError('新しい実験は開始せず、振り返りを実施してください')
    return min(requested, (timestamp(policy['compute_until']) - now).total_seconds())


def collect_review(policy, root):
    root = Path(root)
    rows = []
    for name in policy['evidence']:
        path = root / name
        if not path.is_file():
            rows.append({'path': name, 'status': 'missing'})
            continue
        raw = path.read_bytes()
        rows.append({'path': name, 'status': 'saved',
                     'sha256': hashlib.sha256(raw).hexdigest(), 'result': json.loads(raw)})
    jobs = []
    for job_id in policy.get('job_ids', []):
        path = root / JOBS_ROOT / job_id / 'exit.json'
        jobs.append({'id': job_id, 'status': 'exit_recorded' if path.is_file() else 'exit_receipt_missing',
                     'exit': json.loads(path.read_bytes()) if path.is_file() else None})
    return {'cycle_id': policy['id'], 'performance_achieved': False,
            'agent_review_required': True, 'jobs': jobs, 'evidence': rows,
            'review_questions': [
                '何が改善し、何が改善しなかったか。同条件の比較と未確認事項を区別する。',
                '仮説は支持されたか。反証と進め方の問題を記録する。',
                '継続・方針変更・中断を比較し、次の有限検証の必要性を提案する。',
                '性能未達を維持し、実験終了を性能達成と混同しない。'],
            'next_action': '実プロセスと終了記録を照合し、保存結果から振り返り資料を完成する。新しい計算枠を自動延長しない。'}


def monitor(path):
    path = Path(path).resolve()
    with path.with_suffix('.monitor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(path.with_suffix('.monitor.json'), {
            'started_at': datetime.now(timezone.utc).isoformat(), 'command': sys.argv})
        while True:
            policy = json.loads(path.read_bytes())
            now = datetime.now(timezone.utc)
            phase = phase_at(policy, now)
            if phase != 'compute':
                result = collect_review(policy, Path.cwd())
                result.update(phase=phase, collected_at=now.isoformat())
                atomic_json(path.with_suffix('.review.json'), result)
            if phase == 'review_overdue':
                return
            time.sleep(min(POLL_SECONDS, max(0.01,
                (timestamp(policy['review_until' if phase != 'compute' else 'compute_until']) - now).total_seconds())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('run', 'monitor', 'review'))
    parser.add_argument('--policy', required=True)
    parser.add_argument('--id')
    parser.add_argument('--timeout', type=float)
    arguments = sys.argv[1:]
    separator = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    path = Path(args.policy)
    policy = json.loads(path.read_bytes())
    if args.operation == 'monitor':
        monitor(path)
    elif args.operation == 'review':
        atomic_json(path.with_suffix('.review.json'), collect_review(policy, Path.cwd()))
    else:
        limit = experiment_timeout(policy, datetime.now(timezone.utc), args.timeout)
        print(json.dumps(run_job(JOBS_ROOT, args.id, arguments[separator + 1:], limit)))


if __name__ == '__main__':
    main()
