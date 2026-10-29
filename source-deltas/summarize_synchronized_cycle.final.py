"""同期教材の固定比較を保存出力から整理する。新たな推論はしない。"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_relative_sequence import matching_rows, error_components, SEEDS, SPLITS, SOURCES
from relative_joint_action import file_hash
from resumable_job import atomic_json
from eyedrop import panda_world

JOINTS = 7
MM_PER_M = 1000
PHASES = ('prepare', 'press', 'release', 'retreat')


def main():
    parser = argparse.ArgumentParser()
    for name in ('connection', 'candidate', 'evaluation', 'motion', 'raw', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    connection, candidate, evaluation, motion, raw, out = [Path(getattr(args, name)) for name in
        ('connection', 'candidate', 'evaluation', 'motion', 'raw', 'output')]
    if out.exists():
        raise ValueError('既存の結果を上書きしません')
    hashes = {}

    def read(path):
        hashes[str(path)] = file_hash(path)
        return json.loads(path.read_bytes())

    before_protocol = read(connection/'protocol.json')
    before_quality = read(connection/'result.json')
    train_protocol = read(candidate/'protocol.json')
    trained = read(candidate/'result.json')
    checked = read(evaluation/'result.json')
    moved = read(motion/'result.json')
    for key in ('initial_model_sha256', 'initial_rng_sha256'):
        if before_protocol[key] != train_protocol[key]:
            raise ValueError('学習前の重み・乱数が一致しません')
    for key in ('samples_sha256', 'labels_sha256', 'batches_sha256'):
        if before_quality[key] != train_protocol[key]:
            raise ValueError('比較の教材が一致しません')
    if before_quality['normalization_sha256']['relative'] != train_protocol['normalization_sha256']:
        raise ValueError('比較の尺度が一致しません')
    if not all(value['completed'] for value in (trained, checked, moved)):
        raise ValueError('比較がまだ完了していません')
    if not (trained['model_sha256'] == checked['model_sha256'] == moved['comparison']['finetuned_model_sha256']):
        raise ValueError('検査と全動作の重みが一致しません')
    if trained['weights_fixed_at'] != checked['frozen_before_validation']:
        raise ValueError('検査前の重み固定が確認できません')
    summaries = {mode:{} for mode in ('before', 'after')}
    canonical = {}
    for mode, folder, prefix in (('before', connection, 'before'), ('after', evaluation, 'noise')):
        for split in SPLITS:
            outputs = [read(folder/f'{prefix}-{seed}-{split}.json') for seed in SEEDS]
            summaries[mode][split] = {}
            for source in SOURCES:
                rows = [value[source]['rows'] for value in outputs]
                key = split, source
                for value in rows:
                    if key in canonical:
                        matching_rows(canonical[key], value)
                    canonical[key] = rows[0]
                errors = np.asarray([[np.asarray(row['prediction'])-row['reference'] for row in value] for value in rows])
                metrics = [value[source]['metrics'] for value in outputs]
                summary = dict(error_components(errors),
                    joint_max_abs_rad=float(np.abs(errors[...,:JOINTS]).max()),
                    gap_max_abs_mm=float(np.abs(errors[...,JOINTS]).max()*MM_PER_M),
                    false_press_by_seed=[value.get('false_press') for value in metrics],
                    gap_out_of_range_by_seed=[sum(not panda_world.MIN_GAP_M <= row['prediction'][JOINTS] <= panda_world.MAX_GAP_M for row in value) for value in rows],
                    all_seeds_passed=all(value['passed'] for value in metrics))
                if source == 'sequence':
                    summary['phases'] = {}
                    for phase in PHASES:
                        indices = [i for i,row in enumerate(rows[0]) if row['phase'] == phase]
                        part = errors[:,indices]
                        summary['phases'][phase] = dict(error_components(part),
                            gap_max_abs_mm=float(np.abs(part[...,JOINTS]).max()*MM_PER_M))
                summaries[mode][split][source] = summary
    motion_rows = []
    for row in moved['rows']:
        ident = row['case']
        source = read(raw/ident/'result.json')
        teacher = read(Path(source['original_source'])/'result.json')
        frames = read(raw/ident/'frames.json')
        trace = read(motion/(row['mode']+'-'+ident)/'trace.json')
        pressure = read(motion/(row['mode']+'-'+ident)/'pressure-samples.json')
        commands = [r['action'][JOINTS] for r in trace if r.get('action') is not None]
        actual_gaps = [r['own'][JOINTS] for r in trace if r.get('own') is not None]
        first = trace[0]
        same_initial_state = bool(np.allclose(first['state'], frames[0]['state'], atol=1e-6, rtol=0))
        if not same_initial_state:
            raise ValueError('初期の自己状態が教師と一致しません')
        first_rgb = {}
        for side in ('left', 'right'):
            pair = [raw/ident/f'{side}-000.png', motion/(row['mode']+'-'+ident)/f'{side}-000.png']
            values = [file_hash(p) for p in pair]
            hashes.update({str(p):h for p,h in zip(pair, values, strict=True)})
            first_rgb[side] = dict(teacher_sha256=values[0], model_sha256=values[1], equal=values[0] == values[1])
        if not all(value['equal'] for value in first_rgb.values()):
            raise ValueError('開始時の左右画像が教師と一致しません')
        motion_rows.append(dict(mode=row['mode'], case=ident, accepted=row['accepted'],
            frames=row['frames'], error=row['error'], contacts=len(row['contacts']), drops=row['drops'],
            hits=row['hits'], closed_commands=row['closed_command_samples'],
            closed_pressure=row['closed_positive_pressure_samples'], fluid_audit=row['fluid_audit'],
            domain=row['drop_domain'], retreat_error_m=row['retreat_error_m'],
            final_gap_m=row['final_gap_m'],
            command_gap_range_mm=[min(commands)*MM_PER_M, max(commands)*MM_PER_M] if commands else None,
            actual_gap_range_mm=[min(actual_gaps)*MM_PER_M, max(actual_gaps)*MM_PER_M] if actual_gaps else None,
            max_audit_pressure_kpa=max((r['audit_pressure_kpa'] for r in pressure), default=None),
            completed_simulation_time_s=pressure[-1]['time'] if pressure else 0,
            fixed_teacher_success=source['accepted'], fixed_teacher_drops=teacher['drops'],
            same_initial_state=same_initial_state,
            first_rgb=first_rgb,
            first_action=first.get('action'), first_teacher_action=frames[0]['action']))
    atomic_json(out, dict(completed=True, updates=0, new_inferences=0,
        source_sha256=file_hash(__file__), inputs_sha256=hashes, summaries=summaries, motion=motion_rows,
        action_limits=dict(gap_min_m=panda_world.MIN_GAP_M,gap_max_m=panda_world.MAX_GAP_M,
                           source_sha256=file_hash(panda_world.__file__)),
        training_updates=trained['updates'], fixed_updates_completed=trained['fixed_updates_completed'],
        training_seconds=trained['seconds'], model_sha256=trained['model_sha256'],
        robust_command_precision=checked['robust_BC_passed'],
        learned_motion_passed=moved['finetuned_pilot_passed'],
        adopted=False, independent_people_evaluated=False, agent_review_required=True,
        comparison_scope='same inputs and normalization before versus after this cycle; not isolated loss or data ablation',
        initial_origin=before_protocol['initial_origin'],
        validation_scope=before_protocol['evaluation_scope']))


if __name__ == '__main__':
    main()
