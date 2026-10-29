"""自己観測と実行済み指令だけから押圧履歴を作る。"""
from collections import deque
import math

ACTION_HZ = 20
STEP_S = 1 / ACTION_HZ
OWN_DIM = 16
STATE_DIM = 32
LAG_STEPS = (1, 2, 4, 8, 16, 32)
HISTORY_HORIZON_S = LAG_STEPS[-1] * STEP_S
MAX_GAP_M = 0.08
GAP_TOLERANCE_M = 1e-6
LEGACY_TIMING = 'measured-compression-v1'
COMMAND_TIMING = 'executed-press-command-v2'
TIMING_INDEX = 29
BOTTLE_WIDTH_M = 0.0159


def command_observation_contract():
    return dict(version=2, timing=COMMAND_TIMING, feature_index=TIMING_INDEX,
                bottle_width_m=BOTTLE_WIDTH_M, step_s=STEP_S,
                input='past_executed_gap_commands_only', reset='nonpress_command_or_episode')


def history_from_normalization(document, rest_gap_m):
    contract = document.get('observation_history_contract')
    if contract is None:
        return PressHistory(rest_gap_m)
    if contract != command_observation_contract():
        raise ValueError('未対応または不一致の観測履歴契約です')
    return PressHistory(rest_gap_m, timing=COMMAND_TIMING)


class PressHistory:
    def __init__(self, rest_gap_m, *, timing=LEGACY_TIMING):
        if not math.isfinite(rest_gap_m) or not 0 < rest_gap_m <= MAX_GAP_M:
            raise ValueError("無押圧の指間隔が不正です")
        self.rest_gap_m = rest_gap_m
        if timing not in (LEGACY_TIMING, COMMAND_TIMING):
            raise ValueError('未対応の時間特徴です')
        self.timing = timing
        self.records = deque(maxlen=LAG_STEPS[-1] + 1)

    @staticmethod
    def _validate(own_state, executed_gap_m):
        values = tuple(own_state)
        if len(values) != OWN_DIM or not all(math.isfinite(v) for v in values):
            raise ValueError("自己状態は有限の16成分で指定してください")
        if not 0 <= values[7] <= MAX_GAP_M:
            raise ValueError("実測指間隔が範囲外です")
        if not math.isfinite(executed_gap_m) or not 0 <= executed_gap_m <= MAX_GAP_M:
            raise ValueError("実行済み指令の指間隔が不正です")
        return values

    def reset(self, own_state, initial_gap_command_m):
        values = self._validate(own_state, initial_gap_command_m)
        self.records.clear()
        self.initial = (values, initial_gap_command_m)
        self.records.append(self.initial)
        self.step = 0
        self.closed_distance_m = 0.0
        self.compressed_s = 0.0
        self.press_command_steps = 0
        self.last_release_step = None

    def append(self, step, own_state, executed_gap_m):
        values = self._validate(own_state, executed_gap_m)
        if not self.records:
            raise ValueError("先にresetしてください")
        if type(step) is not int or step != self.step + 1:
            raise ValueError("stepの重複・欠落・逆行は許可しません")
        previous = self.records[-1][0][7]
        gap = values[7]
        was_compressed = previous < self.rest_gap_m - GAP_TOLERANCE_M
        compressed = gap < self.rest_gap_m - GAP_TOLERANCE_M
        self.closed_distance_m += max(previous - gap, 0.0)
        self.compressed_s = self.compressed_s + STEP_S if compressed else 0.0
        self.press_command_steps = self.press_command_steps + 1 if executed_gap_m < BOTTLE_WIDTH_M else 0
        if was_compressed and not compressed:
            self.last_release_step = step
        self.records.append((values, executed_gap_m))
        self.step = step

    def observation(self):
        if not self.records:
            raise ValueError("先にresetしてください")
        records = list(self.records)
        lagged = [records[-1-lag] if lag < len(records) else self.initial
                  for lag in LAG_STEPS]
        since_release = HISTORY_HORIZON_S if self.last_release_step is None else min(
            (self.step-self.last_release_step) * STEP_S, HISTORY_HORIZON_S)
        # 時点tの実行済み指令は直前区間で発行したaction[t-1]であり、次の教師指令ではない。
        commands = [records[-lag][1] if lag <= self.step else self.initial[1]
                    for lag in LAG_STEPS]
        value = [*records[-1][0], *[record[0][7] for record in lagged],
                 *commands, self.closed_distance_m,
                 self.press_command_steps * STEP_S if self.timing == COMMAND_TIMING else self.compressed_s,
                 since_release,
                 sum(lag <= self.step for lag in LAG_STEPS) / len(LAG_STEPS)]
        if len(value) != STATE_DIM:
            raise RuntimeError("履歴の入力次元が契約と一致しません")
        return value
