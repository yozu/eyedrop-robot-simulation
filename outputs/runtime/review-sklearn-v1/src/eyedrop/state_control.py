"""認識値を与えた動作比較。未来の開眼・滴数・液量を入力に含めない。"""
from dataclasses import dataclass
import numpy as np

DT = .05
PRESS_SECONDS = .6
PRESS_MM = 1.
RETREAT = .25
CLEARANCE = .11
MIN_HEIGHT = .16
MAX_HEIGHT = .235
ALIGN = .003
OPEN_THRESHOLD = .95
OPEN_WINDOW = (.15, .35)
APPROACH, PRESS, DONE = range(3)
ORIGIN = np.array([0., -.1, 1.1])


@dataclass
class Memory:
    seen_closed: bool = False
    open_age: float = 0.
    press_age: float = 0.

    def update(self, openness, command):
        self.seen_closed = self.seen_closed or openness < OPEN_THRESHOLD
        self.open_age = round(self.open_age + DT, 8) if openness >= OPEN_THRESHOLD else 0.
        if command > 0:
            self.press_age = round(self.press_age + DT, 8)


def observation(eye_world, own, openness, memory):
    eye = np.array(eye_world)[[0, 2, 1]] - ORIGIN
    row = np.r_[eye - np.asarray(own[:3]), own, openness,
                float(memory.seen_closed), memory.open_age, memory.press_age]
    if row.shape != (11,) or not np.isfinite(row).all() or not 0 <= openness <= 1:
        raise ValueError('観測の形状または値が不正です')
    return row


def goal_for_eye(eye):
    goal = np.asarray(eye).copy()
    goal[..., 2] = np.clip(goal[..., 2] + CLEARANCE, MIN_HEIGHT, MAX_HEIGHT)
    return goal


def control_features(rows, goals):
    rows = np.atleast_2d(rows)
    distance = np.max(np.abs(goals - rows[:, 3:6]), axis=1)
    return np.c_[distance, rows[:, 7:11]]


def rule_modes(features):
    distance, opened, seen, age, pressed = np.asarray(features).T
    start = (distance < ALIGN) & (opened >= OPEN_THRESHOLD) & (seen > 0) & (age >= OPEN_WINDOW[0]) & (age <= OPEN_WINDOW[1])
    active = (pressed > 0) | start
    return np.where(pressed >= PRESS_SECONDS, DONE, np.where(active, PRESS, APPROACH))


def decode(goals, modes):
    actions = np.c_[goals, np.where(modes == PRESS, PRESS_MM, 0.)]
    actions[modes == DONE, 2] = RETREAT
    return actions


def rule_action(row):
    rows = np.atleast_2d(row)
    goals = goal_for_eye(rows[:, :3] + rows[:, 3:6])
    return decode(goals, rule_modes(control_features(rows, goals)))[0].tolist()


class LearnedControl:
    """位置の局所線形回帰と、実演ラベルから学ぶ操作分類器。"""
    def __init__(self, position_tree, coefficients, classifier):
        self.position_tree = position_tree
        self.coefficients = coefficients
        self.classifier = classifier

    def goals(self, eyes):
        eyes = np.atleast_2d(eyes)
        leaves = self.position_tree.apply(eyes[:, 2:3])
        design = np.c_[eyes, np.ones(len(eyes))]
        return np.stack([design[i] @ self.coefficients[int(leaf)] for i, leaf in enumerate(leaves)])

    def actions(self, rows):
        rows = np.atleast_2d(rows)
        goals = self.goals(rows[:, :3] + rows[:, 3:6])
        modes = self.classifier.predict(control_features(rows, goals))
        return decode(goals, modes)

    def action(self, row):
        return self.actions(row)[0].tolist()


class SplineControl(LearnedControl):
    """連続する高さを折れ線基底で回帰する。係数は教師データから推定する。"""
    def goals(self, eyes):
        eyes = np.atleast_2d(eyes)
        design = np.c_[eyes[:, :2], self.position_tree.transform(eyes[:, 2:3])]
        return design @ self.coefficients
