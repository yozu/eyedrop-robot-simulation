"""方策の4軸指令をそのまま実行する環境。眼座標や開眼による自動操作は持たない。"""
import math
import random
import pybullet as pb
from .bottle_world import BottleWorld
from .robot import PHYSICS_HZ

ACTION_HZ = 20
SUBSTEPS = PHYSICS_HZ // ACTION_HZ
ORIGIN = (0., 1.1, -.1)
BOUNDS = ((-.08, .08), (-.16, .12), (.02, .35))
INITIAL_Z = .25
INITIAL_XY = .03
MAX_SQUEEZE = 3.


def scene_point(p):
    return [ORIGIN[0]+p[0], ORIGIN[1]+p[2], ORIGIN[2]+p[1]]


class LearningRun:
    def __init__(self, seed):
        self.seed = seed
        self.frame = 0

    def __enter__(self):
        self.world = BottleWorld(axis_bounds=BOUNDS).__enter__()
        rng = random.Random(self.seed)
        start = [rng.uniform(-INITIAL_XY, INITIAL_XY), rng.uniform(-INITIAL_XY, INITIAL_XY), INITIAL_Z]
        for axis, value in enumerate(start):
            pb.resetJointState(self.world.robot.body, axis, value, physicsClientId=self.world.robot.client)
        self.world.robot.move(start)
        return self

    def snapshot(self):
        snap = self.world.snapshot()
        return {"observation":{"state":[*snap["position"], snap["bottle"]["displacement_mm"]]},
                "time":self.frame/ACTION_HZ, "frame":self.frame,
                "nozzle":scene_point(snap["position"]),
                "drops":[{**d, "position":scene_point(d["position"])} for d in snap["drops"]],
                "bottle":snap["bottle"]}

    def step(self, action):
        bounds = (*BOUNDS,(0.,MAX_SQUEEZE))
        if len(action)!=4 or any(not math.isfinite(v) or not low<=v<=high for v,(low,high) in zip(action,bounds)):
            raise ValueError("4軸指令が可動範囲外です")
        for _ in range(SUBSTEPS):
            self.world.step(action[:3],action[3])
        self.frame += 1
        return self.snapshot()

    def __exit__(self,*args):
        self.world.__exit__(*args)
