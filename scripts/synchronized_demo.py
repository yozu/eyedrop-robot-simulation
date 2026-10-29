"""操作前の観測と次の指令を対応付ける。採点情報は教材列へ含めない。"""
import math

from eyedrop.smolvla_observation import PressHistory, COMMAND_TIMING, ACTION_HZ

STATE_TOLERANCE = 1e-9
ACTION_DIM = 8


class SynchronizedDemo:
    def __init__(self, initial, rest_gap):
        self.history = PressHistory(rest_gap, timing=COMMAND_TIMING)
        self.history.reset(initial, rest_gap)

    def before_action(self, step, own, action):
        state = self.history.observation()
        if step != self.history.step or len(own) != 16 or any(
            not math.isfinite(a) or abs(a-b) > STATE_TOLERANCE for a, b in zip(own, state[:16])
        ):
            raise ValueError('操作前の実測自己状態または時刻が一致しません')
        if len(action) != ACTION_DIM or not all(math.isfinite(a) for a in action):
            raise ValueError('次の指令は有限の8成分が必要です')
        return dict(step=step, time=step/ACTION_HZ, state=state, action=list(action),
                    left=f'left-{step:03}.png', right=f'right-{step:03}.png')

    def after_action(self, step, own, action):
        self.history.append(step+1, own, action[7])
