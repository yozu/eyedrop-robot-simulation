"""学習実演の解除境界を固定する。人物と段階は索引作成だけに使う。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from relative_sequence_data import ACTION_DIM, EPISODE_FRAMES, PRESS_GAP_M

EPISODE_COUNT = 6
MANIFEST_VERSION = 'train-release-pairs-v1'
PAIR_DIFFERENCE = 'endpoint-difference'
PAIR_FLOW_GRID = 'paired-flow-grid'
PAIR_OBJECTIVES = (PAIR_DIFFERENCE, PAIR_FLOW_GRID)
FLOW_STEPS = 10
FLOW_TIMES = tuple(step / FLOW_STEPS for step in range(FLOW_STEPS, 0, -1))


def pair_flow_time(step, pair_count):
    if type(step) is not int or step < 1 or type(pair_count) is not int or pair_count < 1:
        raise ValueError('1から始まる更新番号と正の対数が必要です')
    return FLOW_TIMES[((step - 1) // pair_count) % FLOW_STEPS]


def training_pairs(rows):
    groups, seen = {}, set()
    for row in rows:
        if (row['split'] != 'train' or type(row['episode']) is not int
                or type(row['step']) is not int or not 0 <= row['step'] < EPISODE_FRAMES
                or type(row['dataset_index']) is not int
                or row['dataset_index'] != row['episode'] * EPISODE_FRAMES + row['step']
                or row['dataset_index'] in seen):
            raise ValueError('学習側の元索引・時刻・一意性が必要です')
        reference = row['reference']
        if len(reference) != ACTION_DIM or not all(math.isfinite(v) for v in reference):
            raise ValueError('教師指令の次元または有限性が不正です')
        seen.add(row['dataset_index'])
        groups.setdefault(row['episode'], []).append(row)
    if set(groups) != set(range(EPISODE_COUNT)):
        raise ValueError('固定した学習6実演が必要です')
    pairs = []
    for episode in range(EPISODE_COUNT):
        press = [r for r in groups[episode] if r['phase'] == 'press']
        release = [r for r in groups[episode] if r['phase'] == 'release']
        if not press or not release:
            raise ValueError('押圧と解除の両側が必要です')
        before, after = max(press, key=lambda r:r['step']), min(release, key=lambda r:r['step'])
        if (after['step'] != before['step'] + 1
                or not before['reference'][-1] < PRESS_GAP_M <= after['reference'][-1]):
            raise ValueError('隣接する押圧終了と解除開始が必要です')
        pairs.append(dict(episode=episode, before_index=before['dataset_index'],
                          after_index=after['dataset_index'], before_step=before['step'],
                          after_step=after['step'], before_gap_m=before['reference'][-1],
                          after_gap_m=after['reference'][-1]))
    return pairs


def validate_pair_authorization(parent, objective=PAIR_DIFFERENCE):
    if objective not in PAIR_OBJECTIVES:
        raise ValueError('登録されていない対の学習方式です')
    if parent.get('paired_switch_authorized') is not True:
        raise ValueError('対を使う候補の有限予算が明示されていません')
    if parent.get('paired_switch_objective', PAIR_DIFFERENCE) != objective:
        raise ValueError('承認された対の学習方式と異なります')


def validate_pair_preflight(parent, quality, objective=PAIR_DIFFERENCE):
    validate_pair_authorization(parent, objective)
    if quality.get('paired_switch_ready') is not True:
        raise ValueError('実モデルでの対の損失・勾配・保存復元が未確認です')
    if quality.get('paired_switch_objective', PAIR_DIFFERENCE) != objective:
        raise ValueError('別方式の接続証拠を流用できません')
    if objective == PAIR_FLOW_GRID:
        checks = quality.get('pair_time_checks', [])
        if (tuple(row.get('time') for row in checks) != FLOW_TIMES or
                any(row.get('default_rng_preserved') is not True or
                    not math.isfinite(row.get('loss', float('nan'))) for row in checks)):
            raise ValueError('全10時刻の実モデル接続記録が必要です')


def validate_pair_resume(saved, objective):
    if objective not in PAIR_OBJECTIVES or saved.get('objective', PAIR_DIFFERENCE) != objective:
        raise ValueError('途中保存と対の学習方式が異なります')


def validate_pair_manifest(document, labels, labels_sha256):
    if (document.get('schema') != MANIFEST_VERSION or document.get('source_split') != 'train'
            or document.get('labels_sha256') != labels_sha256
            or document.get('pairs') != training_pairs(labels['train']['sequence'])):
        raise ValueError('対の一覧が固定した学習ラベルと一致しません')
    return document['pairs']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels'); parser.add_argument('output')
    parser.add_argument('--labels-sha256', required=True)
    args = parser.parse_args()
    source = Path(args.labels); raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.labels_sha256:
        raise ValueError('固定した教材ラベルと一致しません')
    pairs = training_pairs(json.loads(raw)['train']['sequence'])
    result = dict(schema=MANIFEST_VERSION, labels_sha256=digest, source_split='train',
                  pairs=pairs, model_loaded=False, updates=0, paired_switch_ready=False)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'pairs':len(pairs), 'updates':0, 'model_loaded':False}))


if __name__ == '__main__':
    main()
