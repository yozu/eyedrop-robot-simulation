"""時間処理の診断モデル。人物メッシュとの衝突判定ではない。"""
import math
from dataclasses import dataclass

GRAVITY = 9.80665
TARGET_RADIUS = 0.006
LID_HEIGHT = 0.001


@dataclass(frozen=True)
class Blink:
    start: float
    closing: float
    closed: float
    opening: float

    def __post_init__(self):
        if self.closing <= 0 or self.closed < 0 or self.opening <= 0:
            raise ValueError("閉開時間が不正です")

    def openness(self, time):
        elapsed = time - self.start
        if elapsed < 0:
            return 1.0
        if elapsed < self.closing:
            return (1 + math.cos(math.pi * elapsed / self.closing)) / 2
        elapsed -= self.closing
        if elapsed < self.closed:
            return 0.0
        elapsed -= self.closed
        if elapsed < self.opening:
            return (1 - math.cos(math.pi * elapsed / self.opening)) / 2
        return 1.0


@dataclass(frozen=True)
class Drop:
    start: float
    x: float
    y: float
    height: float

    def __post_init__(self):
        if self.height <= LID_HEIGHT:
            raise ValueError("放出位置はまぶたより上が必要です")

    def position(self, time):
        elapsed = max(0, time - self.start)
        return self.x, self.y, self.height - GRAVITY * elapsed**2 / 2


def first_hit(drop, previous, current, eye, openness):
    """一更新区間内の平面通過を確認する。眼位置・開きは区間内一定の近似。"""
    if current < previous or not 0 <= openness <= 1:
        raise ValueError("時刻または開眼量が不正です")
    x, y, before = drop.position(previous)
    _, _, after = drop.position(current)
    dx, dy = x - eye[0], y - eye[1]
    in_eye = dx**2 + dy**2 <= TARGET_RADIUS**2
    in_opening = in_eye and abs(dy) < TARGET_RADIUS * openness
    if in_eye and not in_opening and before > LID_HEIGHT >= after:
        return "lid"
    if before > 0 >= after:
        return "target" if in_opening else "face"
    return None
