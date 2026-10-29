"""押してから解除する単体試験。操作方策や実ボトルの成績ではない。"""
from dataclasses import asdict
from .bottle import Bottle, BottleConfig, MODEL_VERSION

TRIAL_HZ = 240
SAMPLE_EVERY = 8
IDLE_S = .25
SETTLE_S = 3.


def trial(target_mm, hold_s, press_speed_mm_s=2., release_speed_mm_s=2.):
    import math
    if not math.isfinite(hold_s) or not .1 <= hold_s <= 5:
        raise ValueError("保持時間は0.1〜5秒です")
    config = BottleConfig(press_speed_mm_s=press_speed_mm_s,
                          release_speed_mm_s=release_speed_mm_s)
    bottle = Bottle(config)
    bottle.validate_command(target_mm, 1 / TRIAL_HZ)
    release_at = IDLE_S + hold_s
    frames, drops = [], []
    total_steps = round((release_at + SETTLE_S) * TRIAL_HZ)
    for tick in range(total_steps):
        time = tick / TRIAL_HZ
        target = target_mm if IDLE_S <= time < release_at else 0.
        drops.extend(asdict(event) for event in bottle.step(target, 1 / TRIAL_HZ))
        if tick % SAMPLE_EVERY == 0 or tick == total_steps - 1:
            frames.append({"target_mm": target, **bottle.snapshot()})
    return {"model": MODEL_VERSION, "config": asdict(config),
            "target_mm": target_mm, "hold_s": hold_s, "release_at_s": release_at,
            "frames": frames, "drops": drops,
            "metrics": {"count": len(drops),
                        "first_drop_s": drops[0]["time_s"] if drops else None,
                        "after_release": sum(d["time_s"] >= release_at for d in drops),
                        "balance_error_ml": config.initial_ml - bottle.reservoir_ml - bottle.tip_ml - bottle.emitted_ml}}
