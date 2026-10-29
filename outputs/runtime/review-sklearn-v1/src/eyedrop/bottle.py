"""押圧・圧力・先端液量を結ぶ簡略モデル。実製品の測定モデルではない。"""
from dataclasses import asdict, dataclass
import math

MODEL_VERSION = "reduced-bottle-v1"
INTEGRATION_STEP_S = 1 / 960
MAX_COMMAND_STEP_S = .1


@dataclass(frozen=True)
class BottleConfig:
    initial_ml: float = 12.
    max_displacement_mm: float = 3.
    press_speed_mm_s: float = 2.
    release_speed_mm_s: float = 2.
    displacement_ml_mm: float = .12
    compliance_ml_kpa: float = .025
    relaxation_s: float = .6
    opening_pressure_kpa: float = 1.
    conductance_ml_s_kpa: float = .08
    drop_ml: float = .03

    def __post_init__(self):
        for name, value in asdict(self).items():
            nonnegative = name in {"initial_ml", "opening_pressure_kpa"}
            if not math.isfinite(value) or (value < 0 if nonnegative else value <= 0):
                raise ValueError(f"容器設定が不正です: {name}")


@dataclass(frozen=True)
class DropEvent:
    time_s: float
    volume_ml: float


class Bottle:
    def __init__(self, config=None):
        self.config = config or BottleConfig()
        self.time_s = 0.
        self.displacement_mm = 0.
        self.pressure_kpa = 0.
        self.reservoir_ml = self.config.initial_ml
        self.tip_ml = 0.
        self.emitted_ml = 0.
        self.drop_count = 0

    def snapshot(self):
        return {"model": MODEL_VERSION, "time_s": self.time_s,
                "displacement_mm": self.displacement_mm, "pressure_kpa": self.pressure_kpa,
                "reservoir_ml": self.reservoir_ml, "tip_ml": self.tip_ml,
                "emitted_ml": self.emitted_ml, "drop_count": self.drop_count}

    def validate_command(self, target_mm, dt_s):
        c = self.config
        if not math.isfinite(target_mm) or not 0 <= target_mm <= c.max_displacement_mm:
            raise ValueError("押圧目標が可動範囲外です")
        if not math.isfinite(dt_s) or not 0 < dt_s <= MAX_COMMAND_STEP_S:
            raise ValueError("更新時間が不正です")
    def step(self, target_mm, dt_s):
        return self._advance(target_mm, dt_s, measured=False)

    def step_measured(self, displacement_mm, dt_s):
        # 物理エンジンの実測変位には指令用の速度制限を重ねない。
        return self._advance(displacement_mm, dt_s, measured=True)

    def _advance(self, target_mm, dt_s, *, measured):
        self.validate_command(target_mm, dt_s)
        c = self.config
        count = math.ceil(dt_s / INTEGRATION_STEP_S)
        h = dt_s / count
        measured_delta = (target_mm - self.displacement_mm) / count
        events = []
        for _ in range(count):
            distance = target_mm - self.displacement_mm
            speed = c.press_speed_mm_s if distance >= 0 else c.release_speed_mm_s
            delta = measured_delta if measured else math.copysign(min(abs(distance), speed * h), distance)
            self.displacement_mm += delta
            # 圧縮による圧力増分、容器・空気系の緩和、ノズル流出の順に積分する。
            self.pressure_kpa += c.displacement_ml_mm * delta / c.compliance_ml_kpa
            self.pressure_kpa *= math.exp(-h / c.relaxation_s)
            flow = c.conductance_ml_s_kpa * max(0., self.pressure_kpa - c.opening_pressure_kpa)
            volume = min(flow * h, self.reservoir_ml,
                         max(0., self.pressure_kpa) * c.compliance_ml_kpa)
            self.pressure_kpa -= volume / c.compliance_ml_kpa
            self.reservoir_ml -= volume
            self.tip_ml += volume
            self.time_s += h
            # 解除しても先端の液量は消さない。再蓄積すれば複数滴が生成される。
            while self.tip_ml >= c.drop_ml:
                self.tip_ml -= c.drop_ml
                self.emitted_ml += c.drop_ml
                self.drop_count += 1
                events.append(DropEvent(self.time_s, c.drop_ml))
        return events
