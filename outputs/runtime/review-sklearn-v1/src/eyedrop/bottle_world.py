"""学習用容器の滴下イベントを3軸機構と重力粒子へ接続する。採点は別処理。"""
import math
import pybullet as pb
from .bottle import Bottle
from .robot import RobotWorld, PHYSICS_HZ, LINK_SIZE, AXIS_BOUNDS

ML_TO_M3 = 1e-6
DENSITY_KG_M3 = 1000.
SPHERE_VOLUME_FACTOR = 4 * math.pi / 3
SPAWN_CLEARANCE_M = .0001


class BottleWorld:
    def __init__(self, config=None, axis_bounds=AXIS_BOUNDS):
        self.bottle = Bottle(config)
        self.particles = []
        self.axis_bounds = axis_bounds

    def __enter__(self):
        self.robot = RobotWorld(self.axis_bounds).__enter__()
        return self

    def step(self, position_target, squeeze_target_mm):
        dt = 1 / PHYSICS_HZ
        self.bottle.validate_command(squeeze_target_mm, dt)
        self.robot.move(position_target)
        events = self.bottle.step(squeeze_target_mm, dt)
        self.robot.step()
        for event in events:
            volume_m3 = event.volume_ml * ML_TO_M3
            radius = (volume_m3 / SPHERE_VOLUME_FACTOR) ** (1 / 3)
            shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius,
                                            physicsClientId=self.robot.client)
            position = list(self.robot.position())
            position[2] -= LINK_SIZE + radius + SPAWN_CLEARANCE_M
            body = pb.createMultiBody(baseMass=volume_m3 * DENSITY_KG_M3,
                                       baseCollisionShapeIndex=shape, basePosition=position,
                                       physicsClientId=self.robot.client)
            # 原点に置かれた水平軸の仮形状を液滴経路から除外する。
            for axis in (0, 1):
                pb.setCollisionFilterPair(body, self.robot.body, -1, axis, 0,
                                          physicsClientId=self.robot.client)
            pb.changeDynamics(body, -1, linearDamping=0, angularDamping=0,
                              physicsClientId=self.robot.client)
            self.particles.append({"body": body, "volume_ml": event.volume_ml,
                                   "radius_m": radius, "released_s": event.time_s})
        return self.snapshot()

    def snapshot(self):
        drops = [{"id": i, "volume_ml": d["volume_ml"], "radius_m": d["radius_m"],
                  "released_s": d["released_s"], "position": pb.getBasePositionAndOrientation(
                      d["body"], physicsClientId=self.robot.client)[0]}
                 for i, d in enumerate(self.particles)]
        return {"bottle": self.bottle.snapshot(), "position": self.robot.position(), "drops": drops}

    def __exit__(self, *args):
        self.robot.__exit__(*args)
