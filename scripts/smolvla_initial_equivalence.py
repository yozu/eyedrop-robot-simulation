"""LoRA統合の数値差と、操作の精度基準を区別する。"""
import numpy as np

MICRO_CASES = 9
ACTION_DIMENSIONS = 8
INITIAL_JOINT_LIMIT_RAD = 1e-5
INITIAL_GAP_LIMIT_M = 1e-7


def validate_initial_equivalence(reference, actual):
    if len(reference) != MICRO_CASES or len(actual) != MICRO_CASES:
        raise ValueError('同じ9例の初期比較が必要です')
    for left, right in zip(reference, actual):
        if left['index'] != right['index'] or left['actual_open'] != right['actual_open'] or left['reference'] != right['reference']:
            raise ValueError('行順、条件、教師が異なります')
    a = np.asarray([row['prediction'] for row in reference], dtype=np.float64)
    b = np.asarray([row['prediction'] for row in actual], dtype=np.float64)
    if a.shape != (MICRO_CASES, ACTION_DIMENSIONS) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('初期出力の形または有限性が不正です')
    difference = np.abs(a - b)
    joint = float(difference[:, :7].max())
    gap = float(difference[:, 7].max())
    if joint > INITIAL_JOINT_LIMIT_RAD or gap > INITIAL_GAP_LIMIT_M:
        raise ValueError(f'初期予測が許容する丸め差を超えています: joint={joint},gap={gap}')
    return dict(passed=True, bit_equal=bool(np.array_equal(a, b)), joint_max_difference_rad=joint,
                gap_max_difference_m=gap, joint_limit_rad=INITIAL_JOINT_LIMIT_RAD, gap_limit_m=INITIAL_GAP_LIMIT_M)
