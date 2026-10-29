"""相対動作比較用の教材抽出と採点。モデル入力へ教師情報を渡さない。"""
import math
import random

PHASES = ('prepare', 'press', 'release', 'retreat')
MOVING_PHASES = ('prepare', 'retreat')
PHASE_SAMPLE_CAP = 16
EPISODE_FRAMES = 1200
FPS = 20
STATE_DIM = 32
OWN_DIM = 16
ACTION_DIM = 8
JOINT_DIM = 7
MICRO_COUNT = 9
SOURCE_TOLERANCE = 1e-9
NONZERO_RAD = 1e-6
JOINT_RMSE_LIMIT_RAD = .001
JOINT_MAX_LIMIT_RAD = .005
GAP_MAX_LIMIT_M = .00025
PRESS_GAP_M = .0158
OPEN_MINIMUM = .95
MANIFEST_VERSION = 'relative-sequence-phase-cap-v1'


def _vector(value, length):
    if len(value) != length or not all(math.isfinite(v) for v in value):
        raise ValueError('列の次元または有限値の契約が不正です')
    return list(value)


def _same(a, b):
    return len(a) == len(b) and all(abs(x - y) <= SOURCE_TOLERANCE for x, y in zip(a, b))


def _even_steps(steps):
    # 短い区間は全件を使い、長い区間は両端を含めて等間隔に選ぶ。
    count = min(len(steps), PHASE_SAMPLE_CAP)
    if count == 0:
        raise ValueError('教師に必要な動作段階がありません')
    if count == 1:
        return steps[:]
    return [steps[i * (len(steps) - 1) // (count - 1)] for i in range(count)]


def sequence_rows(frames, trace, *, split, episode, case_id, expected_frames=EPISODE_FRAMES):
    if split not in ('train', 'validation') or type(episode) is not int or episode < 0:
        raise ValueError('分割または実演番号が不正です')
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError('実演長は正の整数で指定してください')
    if len(frames) != expected_frames or len(trace) != expected_frames:
        raise ValueError('観測と元traceの長さが異なります')
    groups = {phase: [] for phase in PHASES}
    for i, (frame, record) in enumerate(zip(frames, trace)):
        if frame['step'] != i or record['step'] != i or not math.isfinite(frame['time']) or abs(frame['time'] - i / FPS) > SOURCE_TOLERANCE:
            raise ValueError('画像・自己状態・指令の時刻対応が不正です')
        state = _vector(frame['state'], STATE_DIM)
        action = _vector(frame['action'], ACTION_DIM)
        original = _vector(record['action'], ACTION_DIM)
        _vector(record['own'], OWN_DIM)
        if not _same(action, original):
            raise ValueError('回収指令が元traceと一致しません')
        if i and not _same(state[:OWN_DIM], trace[i - 1]['own']):
            raise ValueError('操作前状態が直前の操作後状態と一致しません')
        phase = record['decision']['phase']
        if phase not in groups:
            raise ValueError('未知の動作段階を検出しました')
        opening = record['audit_open']
        if not math.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError('採点用開眼度が不正です')
        groups[phase].append(i)
    rows = []
    for phase, steps in groups.items():
        selected = _even_steps(steps)
        if phase in MOVING_PHASES and not any(
            max(abs(a - q) for a, q in zip(frames[i]['action'][:JOINT_DIM], frames[i]['state'][:JOINT_DIM])) > NONZERO_RAD
            for i in selected
        ):
            raise ValueError('抽出した接近または退避に非ゼロ移動がありません')
        for i in selected:
            frame, record = frames[i], trace[i]
            rows.append(dict(split=split, episode=episode, case_id=case_id, step=i,
                             dataset_index=episode * expected_frames + i, phase=phase,
                             actual_open=record['audit_open'], observed_q=list(frame['state'][:JOINT_DIM]),
                             reference=list(frame['action']), left=frame['left'], right=frame['right']))
    return rows


def mixed_batches(rows, *, updates, batch_size, seed):
    if any(type(v) is not int or v <= 0 for v in (updates, batch_size)):
        raise ValueError('更新数とバッチ数は正の整数で指定してください')
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('学習splitの教材だけを抽選できます')
    groups = {}
    seen = set()
    for row in rows:
        key = (row['episode'], row['phase'])
        index = row['dataset_index']
        if row['phase'] not in PHASES or type(index) is not int or index < 0 or index in seen:
            raise ValueError('教材の段階・索引・重複が不正です')
        seen.add(index)
        groups.setdefault(key, []).append(index)
    episodes = sorted({episode for episode, _ in groups})
    if set(episodes) != set(range(len(episodes))) or any((episode, phase) not in groups for episode in episodes for phase in PHASES):
        raise ValueError('全実演の4段階が必要です')
    rng = random.Random(seed)
    micro_order = list(range(MICRO_COUNT))
    group_order = [(episode, phase) for episode in episodes for phase in PHASES]
    rng.shuffle(micro_order)
    rng.shuffle(group_order)
    batches = []
    for update in range(updates):
        batch = []
        for slot in range(batch_size):
            draw = update * batch_size + slot
            if draw % 2 == 0:
                batch.append(dict(source='micro', index=micro_order[(draw // 2) % MICRO_COUNT]))
            else:
                key = group_order[(draw // 2) % len(group_order)]
                batch.append(dict(source='sequence', index=rng.choice(groups[key]), episode=key[0], phase=key[1]))
        batches.append(batch)
    return batches


def _metrics(rows):
    if not rows:
        return dict(count=0, passed=False)
    joint_errors, gap_errors, zero_errors = [], [], []
    press_count = press_match = nonpress_count = nonpress_match = false_press = teacher_closed_press = 0
    for row in rows:
        prediction = _vector(row['prediction'], ACTION_DIM)
        reference = _vector(row['reference'], ACTION_DIM)
        q = _vector(row['observed_q'], JOINT_DIM)
        opening = row['actual_open']
        if not math.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError('採点用開眼度が不正です')
        joint_errors.extend(a - b for a, b in zip(prediction[:JOINT_DIM], reference[:JOINT_DIM]))
        zero_errors.extend(a - b for a, b in zip(q, reference[:JOINT_DIM]))
        gap_error = abs(prediction[JOINT_DIM] - reference[JOINT_DIM])
        gap_errors.append(gap_error)
        teacher_press = reference[JOINT_DIM] < PRESS_GAP_M
        predicted_press = prediction[JOINT_DIM] < PRESS_GAP_M
        if teacher_press:
            press_count += 1
            press_match += predicted_press and gap_error <= GAP_MAX_LIMIT_M
        else:
            nonpress_count += 1
            nonpress_match += not predicted_press and gap_error <= GAP_MAX_LIMIT_M
        if opening < OPEN_MINIMUM:
            false_press += predicted_press
            teacher_closed_press += teacher_press
    rmse = math.sqrt(sum(v * v for v in joint_errors) / len(joint_errors))
    joint_max = max(abs(v) for v in joint_errors)
    gap_max = max(gap_errors)
    passed = (rmse <= JOINT_RMSE_LIMIT_RAD and joint_max <= JOINT_MAX_LIMIT_RAD and gap_max <= GAP_MAX_LIMIT_M
              and press_match == press_count and nonpress_match == nonpress_count and false_press == 0 and teacher_closed_press == 0)
    return dict(count=len(rows), passed=passed, joint_rmse_rad=rmse, joint_max_abs_rad=joint_max,
                gap_max_abs_mm=gap_max * 1000, press_reference_count=press_count, press_match=press_match,
                nonpress_reference_count=nonpress_count, nonpress_match=nonpress_match,
                false_press=false_press, teacher_closed_press=teacher_closed_press,
                zero_joint_rmse_rad=math.sqrt(sum(v * v for v in zero_errors) / len(zero_errors)))


def command_metrics(rows):
    if any(row['phase'] not in PHASES for row in rows):
        raise ValueError('未知の動作段階を採点できません')
    result = _metrics(rows)
    result['phases'] = {phase: _metrics([r for r in rows if r['phase'] == phase]) for phase in PHASES}
    moving, holding = [], []
    for row in rows:
        motion = max(abs(a - q) for a, q in zip(_vector(row['reference'], ACTION_DIM)[:JOINT_DIM], _vector(row['observed_q'], JOINT_DIM))) > NONZERO_RAD
        (moving if motion else holding).append(row)
    result['motion'], result['hold'] = _metrics(moving), _metrics(holding)
    result['passed'] = bool(result['passed'] and all(v['passed'] for v in result['phases'].values())
                            and result['motion']['passed']
                            and result['motion']['joint_rmse_rad'] < result['motion']['zero_joint_rmse_rad'])
    result['scope'] = '単時点教師指令との一致。点眼全動作・残圧・非接触の証明ではない'
    return result
