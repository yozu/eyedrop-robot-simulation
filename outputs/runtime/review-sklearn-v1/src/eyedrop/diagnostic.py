"""正解状態を使う描画・物理接続診断。AI方式の比較成績には含めない。"""
import math
import pybullet as pb
from .robot import RobotWorld, PHYSICS_HZ
from .motion import Blink, TARGET_RADIUS

DISPLAY_HZ = 30
STEPS_PER_FRAME = PHYSICS_HZ // DISPLAY_HZ
APPROACH_HEIGHT = .08
DISPENSE_HEIGHT = .02
MIN_DISPENSE_HEIGHT = .02
ALIGN_TOLERANCE = .001
STABLE_SECONDS = .1
SQUEEZE_SECONDS = .3
OPEN_THRESHOLD = .9
BLINK_CYCLE = 9
BLINKS = tuple(Blink(t, .095, .04, .17) for t in (1.6, 3.65, 4.12, 7.3))


def to_scene(position, origin):
    """物理のZ上向きを、描画のY上向きへ変換する。単位は双方メートル。"""
    return (origin[0] + position[0], origin[1] + position[2], origin[2] + position[1])


class DiagnosticRun:
    def __init__(self, origin, dispense_height=DISPENSE_HEIGHT):
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("目の基準位置が不正です")
        self.origin = tuple(origin)
        if not math.isfinite(dispense_height) or not MIN_DISPENSE_HEIGHT <= dispense_height <= APPROACH_HEIGHT:
            raise ValueError("点眼高さが範囲外です")
        self.dispense_height = dispense_height
        self.tick = 0
        self.open_since = None
        self.openness = 1.
        self.status = "moving"
        self.result = None
        self.released = 0
        self.squeeze = 0.

    def __enter__(self):
        self.world = RobotWorld().__enter__()
        pb.resetJointState(self.world.body, 2, APPROACH_HEIGHT, physicsClientId=self.world.client)
        self.world.move((0, 0, self.dispense_height))
        return self

    def snapshot(self):
        return {"time": self.tick / PHYSICS_HZ, "openness": self.openness,
                "nozzle": to_scene(self.world.position(), self.origin),
                "drop": to_scene(self.world.drop_position(), self.origin) if self.released else None,
                "squeeze": self.squeeze, "status": self.status, "result": self.result, "released": self.released,
                "mode": "diagnostic-oracle", "eye": self.origin,"dispense_height":self.dispense_height}

    def advance(self, force_closed=False):
        if self.result is not None:
            return self.snapshot()
        for _ in range(STEPS_PER_FRAME):
            previous = self.world.drop_position() if self.released else None
            self.world.step()
            self.tick += 1
            time = self.tick / PHYSICS_HZ
            self.openness = 0. if force_closed else min(b.openness(time % BLINK_CYCLE) for b in BLINKS)
            x, y, z = self.world.position()
            aligned = max(abs(x), abs(y), abs(z - self.dispense_height)) <= ALIGN_TOLERANCE
            if not self.released:
                good = aligned and self.openness >= OPEN_THRESHOLD
                if not good:
                    self.open_since = None
                    self.squeeze = 0.
                elif self.open_since is None:
                    self.open_since = time
                self.status = "waiting" if aligned else "moving"
                if self.open_since is not None and time - self.open_since >= STABLE_SECONDS:
                    self.squeeze = min(1., (time - self.open_since - STABLE_SECONDS) / SQUEEZE_SECONDS)
                    self.status = "squeezing"
                    if self.squeeze >= 1:
                        self.world.release()
                        self.released = 1
                        self.status = "falling"
            elif previous is not None:
                current = self.world.drop_position()
                if previous[2] > 0 >= current[2]:
                    inside = current[0]**2 + current[1]**2 <= TARGET_RADIUS**2
                    self.result = ("target" if self.openness >= OPEN_THRESHOLD else "lid") if inside else "face"
                    self.status = "finished"
                    break
        return self.snapshot()

    def __exit__(self, *args):
        self.world.__exit__(*args)
