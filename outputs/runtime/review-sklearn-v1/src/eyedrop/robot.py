"""3軸直動機構の診断用物理環境。人物との統合は未実施。"""
import math
import pybullet as pb
from .motion import GRAVITY

PHYSICS_HZ = 240
MAX_SPEED = 0.02
MOTOR_FORCE = 10
AXIS_BOUNDS = ((-0.05, 0.05), (-0.05, 0.05), (0.02, 0.08))
INITIAL_HEIGHT = 0.04
DROP_RADIUS = 0.0005
DROP_MASS = 0.00003
LINK_MASS = 0.1
LINK_SIZE = 0.002


class RobotWorld:
    def __init__(self, axis_bounds=AXIS_BOUNDS):
        self.axis_bounds = axis_bounds

    def __enter__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -GRAVITY, physicsClientId=self.client)
        pb.setTimeStep(1 / PHYSICS_HZ, physicsClientId=self.client)
        shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[LINK_SIZE]*3,
                                        physicsClientId=self.client)
        self.body = pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            linkMasses=[LINK_MASS]*3, linkCollisionShapeIndices=[shape]*3,
            linkVisualShapeIndices=[-1]*3, linkPositions=[[0, 0, 0]]*3,
            linkOrientations=[[0, 0, 0, 1]]*3,
            linkInertialFramePositions=[[0, 0, 0]]*3,
            linkInertialFrameOrientations=[[0, 0, 0, 1]]*3,
            linkParentIndices=[0, 1, 2],
            linkJointTypes=[pb.JOINT_PRISMATIC]*3,
            linkJointAxis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            physicsClientId=self.client)
        pb.resetJointState(self.body, 2, INITIAL_HEIGHT, physicsClientId=self.client)
        self.drop = None
        self.move((0, 0, INITIAL_HEIGHT))
        return self

    def move(self, target):
        if len(target) != len(self.axis_bounds) or any(
            not math.isfinite(value) or not low <= value <= high
            for value, (low, high) in zip(target, self.axis_bounds)
        ):
            raise ValueError("3軸の可動範囲外です")
        for axis, value in enumerate(target):
            pb.setJointMotorControl2(self.body, axis, pb.POSITION_CONTROL,
                                     targetPosition=value, maxVelocity=MAX_SPEED,
                                     force=MOTOR_FORCE, physicsClientId=self.client)

    def step(self):
        pb.stepSimulation(physicsClientId=self.client)

    def position(self):
        return tuple(pb.getJointState(self.body, axis, physicsClientId=self.client)[0]
                     for axis in range(3))

    def release(self):
        if self.drop is not None:
            raise RuntimeError("1エピソード最大1滴です")
        shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=DROP_RADIUS,
                                        physicsClientId=self.client)
        position = list(self.position())
        position[2] -= LINK_SIZE + DROP_RADIUS
        self.drop = pb.createMultiBody(baseMass=DROP_MASS, baseCollisionShapeIndex=shape,
                                       basePosition=position, physicsClientId=self.client)
        # 水平軸の仮形状は原点にあるため、粒子経路の障害物としては扱わない。
        # 実際のレール配置・顔との衝突形状は別途構築する。
        for axis in (0, 1):
            pb.setCollisionFilterPair(self.drop, self.body, -1, axis, 0,
                                      physicsClientId=self.client)
        pb.changeDynamics(self.drop, -1, linearDamping=0, angularDamping=0,
                           physicsClientId=self.client)

    def drop_position(self):
        if self.drop is None:
            raise RuntimeError("まだ放出していません")
        return pb.getBasePositionAndOrientation(self.drop, physicsClientId=self.client)[0]

    def __exit__(self, *_):
        pb.disconnect(self.client)
