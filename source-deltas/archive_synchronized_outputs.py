"""終了した同期教材比較を、非圧縮の保存束と照合用ハッシュにまとめる。"""
import json
import os
from pathlib import Path
import platform
import subprocess

from relative_joint_action import file_hash
from resumable_job import atomic_json

OUTPUT = Path('outputs/archives/synchronized-cycle-20261027-01')
CYCLE = Path('outputs/diagnostics/smolvla-synchronized-cycle-02')
FOLDERS = (
    'outputs/learning/panda-synchronized-data-01',
    'outputs/learning/panda-synchronized-dataset-01',
    'outputs/diagnostics/smolvla-synchronized-connection-02',
    'outputs/diagnostics/smolvla-synchronized-pair-01',
    'outputs/learning/smolvla-synchronized-01',
    'outputs/diagnostics/smolvla-synchronized-check-01',
    'outputs/diagnostics/smolvla-synchronized-motion-01',
    'outputs/diagnostics/smolvla-official-initial-01',
    'outputs/diagnostics/smolvla-official-rng-start-01',
    'outputs/runtime/review-sklearn-v1',
    str(CYCLE/'source-snapshot'),
    str(CYCLE/'source-deltas'),
)


def main():
    if platform.system() != 'Darwin':
        raise ValueError('Mac上で保存します')
    if not json.loads((CYCLE/'result.json').read_bytes())['completed']:
        raise ValueError('比較工程が未完了です')
    policy_path = Path('experiments/autonomous-resume/synchronized-cycle-20261027-01.json')
    policy = json.loads(policy_path.read_bytes())
    paths = list(FOLDERS) + [str(policy_path), 'scripts/archive_synchronized_outputs.py']
    paths.extend(str(p) for p in CYCLE.glob('*.json'))
    for ident in policy['job_ids']:
        folder = Path('outputs/autonomous-jobs')/ident
        receipt = json.loads((folder/'exit.json').read_bytes())
        if receipt.get('live_descendants'):
            raise ValueError('保存対象の処理が残っています')
        paths.append(str(folder))
    for name in paths:
        if not Path(name).exists():
            raise FileNotFoundError(name)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = sorted({f for name in paths for f in
        ([Path(name)] if Path(name).is_file() else Path(name).rglob('*')) if f.is_file()})
    manifest = OUTPUT/'files.json'
    atomic_json(manifest, dict(files=[dict(path=str(p), bytes=p.stat().st_size) for p in files],
        scope='current recorded data, fixed inputs, initial/final weights, all candidate checkpoints, evaluation, runtime and job receipts',
        prerequisites='repository source, licensed person assets and original teacher dependencies remain separate',
        omitted_duplicate='failed connection01 samples match connection02; original failure retained separately in private Git'))
    archive = OUTPUT/'artifacts.tar'
    partial = OUTPUT/'artifacts.tar.partial'
    subprocess.run(['/usr/bin/tar', '-cf', str(partial), '--', *paths, str(manifest)], check=True)
    os.replace(partial, archive)
    atomic_json(OUTPUT/'result.json', dict(completed=True, archive=str(archive),
        bytes=archive.stat().st_size, sha256=file_hash(archive),
        files=len(files), source_bytes=sum(p.stat().st_size for p in files),
        manifest_sha256=file_hash(manifest), code_sha256=file_hash(__file__),
        scope='storage only; no training, inference or physics'))
    print((OUTPUT/'result.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
