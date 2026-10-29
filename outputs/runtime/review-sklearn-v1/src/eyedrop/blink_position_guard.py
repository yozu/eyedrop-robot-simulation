"""短いまばたき中は、直前に画像で確認した位置だけを保持する。"""
import math

OPEN_THRESHOLD = .95
MAX_HOLD_SECONDS = .35


class BlinkPositionGuard:
    def __init__(self):
        self.last_point = None
        self.last_open_time = None
        self.previous_time = None

    def update(self, point, opening, time_s):
        if (len(point) != 3 or not all(math.isfinite(v) for v in point)
                or not math.isfinite(opening) or not 0<=opening<=1
                or not math.isfinite(time_s) or time_s<0
                or self.previous_time is not None and time_s<self.previous_time):
            raise ValueError('画像位置・開眼度・時刻が不正です')
        self.previous_time = time_s
        if opening>=OPEN_THRESHOLD:
            self.last_point = list(point)
            self.last_open_time = time_s
        if self.last_open_time is None or time_s-self.last_open_time>MAX_HOLD_SECONDS:
            return None
        return self.last_point.copy()
