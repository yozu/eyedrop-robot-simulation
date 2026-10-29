"""Pandaの関節運動・容器・全リンク接触を扱うMac用試験環境。"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
import pybullet_data
from .bottle import Bottle
from .drop_kinematics import LEGACY, MOVING_NOZZLE, nozzle_motion, emit_drop, advance_drop, domain_sample

PHYSICS_HZ = 240
ACTION_HZ = 20
SUBSTEPS = PHYSICS_HZ // ACTION_HZ
CONTACT_MARGIN_M = .001
ARM_SPEED_RAD_S = .3
FINGER_SPEED_M_S = .002
PHYSICS_JOINT_VELOCITY_CAP = .3
FINGER_TRAVEL_BOUND_M_S = PHYSICS_JOINT_VELOCITY_CAP
REST_GAP_M = .0199
BOTTLE_DEPTH_M = .0159
NOZZLE_LOCAL = (0, 0, .042)
TOOL_OFFSET_M = .015
NOZZLE_GRASP_OFFSET_M = TOOL_OFFSET_M + NOZZLE_LOCAL[2]
HOME = (0, -.5, 0, -2.2, 0, 1.8, .8)
DEFAULT_BASE = (.48, -.16, .80)
MESH_CHUNK_INDICES = 6000
BLINK_PADDING_BIN_M = .001
GRAVITY_M_S2 = 9.81
GAP_TOLERANCE_M = .0001
MIN_GAP_M = .0129
MAX_GAP_M = .08
ARM_JOINT_COUNT=7


def synchronized_joint_target(current,desired,max_delta,limits):
    current=np.asarray(current,dtype=float);desired=np.asarray(desired,dtype=float);limits=np.asarray(limits,dtype=float)
    if (current.shape!=(ARM_JOINT_COUNT,) or desired.shape!=(ARM_JOINT_COUNT,) or limits.shape!=(ARM_JOINT_COUNT,2)
            or not np.isfinite(current).all() or not np.isfinite(desired).all() or not np.isfinite(limits).all()
            or not math.isfinite(max_delta) or max_delta<=0 or (limits[:,0]>limits[:,1]).any()):
        raise ValueError('同期させる関節指令が不正です')
    if ((current<limits[:,0])|(current>limits[:,1])|(desired<limits[:,0])|(desired>limits[:,1])).any():
        raise ValueError('元の関節目標が可動域外です。制限で隠しません')
    delta=desired-current;largest=float(np.max(abs(delta)))
    if largest<=max_delta:return desired.tolist()
    return (current+delta*(max_delta/largest)).tolist()


def to_scene(point):
    return [point[0], point[2], -point[1]]


def to_physics(point):
    return [point[0], -point[2], point[1]]


def swept_margin(joint_angles, reach_bounds, prismatic_m=0):
    angles = np.asarray(joint_angles, dtype=float)
    radii = np.asarray(reach_bounds, dtype=float)
    if angles.shape != radii.shape or not np.isfinite(angles).all() or not np.isfinite(radii).all():
        raise ValueError("表面移動上限が不明です")
    if (radii < 0).any() or not math.isfinite(prismatic_m) or prismatic_m < 0:
        raise ValueError("表面移動上限が不正です")
    return float(np.abs(angles) @ radii + prismatic_m)


def make_urdf(destination):
    assets = Path(pybullet_data.getDataPath()) / "franka_panda"
    tree = ET.parse(assets / "panda.urdf")
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str(assets / mesh.get("filename").replace("package://", "")))
    tool = ET.SubElement(root, "link", name="eyedrop_bottle")
    inertial = ET.SubElement(tool, "inertial")
    ET.SubElement(inertial, "mass", value="0.025")
    ET.SubElement(inertial, "inertia", ixx="0.00001", ixy="0", ixz="0", iyy="0.00001", iyz="0", izz="0.00001")
    for xyz, size, rgba in [("0 0 0", ".0393 .0159 .0497", ".2 .65 .8 1"),
                            ("0 0 .028", ".008 .008 .012", ".95 .95 .95 1"),
                            ("0 0 .038", ".003 .003 .008", ".95 .95 .95 1")]:
        for kind in ("visual", "collision"):
            shape = ET.SubElement(tool, kind)
            ET.SubElement(shape, "origin", xyz=xyz)
            ET.SubElement(ET.SubElement(shape, "geometry"), "box", size=size)
            if kind == "visual":
                ET.SubElement(ET.SubElement(shape, "material", name="bottle"), "color", rgba=rgba)
    joint = ET.SubElement(root, "joint", name="bottle_mount", type="fixed")
    ET.SubElement(joint, "parent", link="panda_grasptarget")
    ET.SubElement(joint, "child", link="eyedrop_bottle")
    ET.SubElement(joint, "origin", xyz=f"0 0 {TOOL_OFFSET_M}")
    for side,sign in (("left",1),("right",-1)):
        pad=ET.SubElement(root,"link",name=f"{side}_bottle_pad")
        inertial=ET.SubElement(pad,"inertial")
        ET.SubElement(inertial,"mass",value=".01")
        ET.SubElement(inertial,"inertia",ixx=".00001",ixy="0",ixz="0",iyy=".00001",iyz="0",izz=".00001")
        for kind in ('visual','collision'):
            item=ET.SubElement(pad,kind)
            ET.SubElement(ET.SubElement(item,'geometry'),'box',size='.012 .008 .06')
        joint=ET.SubElement(root,'joint',name=f'{side}_pad_mount',type='fixed')
        ET.SubElement(joint,'parent',link=f'panda_{side}finger')
        ET.SubElement(joint,'child',link=f'{side}_bottle_pad')
        ET.SubElement(joint,'origin',xyz=f'0 {sign*.004} .0616')
    tree.write(destination)


class PandaWorld:
    drop_model=LEGACY

    def __init__(self, directory, base=DEFAULT_BASE,rest_gap=REST_GAP_M, *, bottle_input='command',finger_release_speed_m_s=FINGER_SPEED_M_S,finger_press_speed_m_s=FINGER_SPEED_M_S,joint_velocity_cap=PHYSICS_JOINT_VELOCITY_CAP,drop_model=LEGACY):
        if drop_model not in (LEGACY,MOVING_NOZZLE):raise ValueError('滴の運動モデルを明示してください')
        self.drop_model=drop_model
        if not math.isfinite(joint_velocity_cap) or not 0<joint_velocity_cap<=PHYSICS_JOINT_VELOCITY_CAP:
            raise ValueError('物理速度上限が既存の保証範囲外です')
        if bottle_input not in ('command','measured'):
            raise ValueError('容器の入力方式が不正です')
        if not math.isfinite(finger_release_speed_m_s) or not 0<finger_release_speed_m_s<=joint_velocity_cap:
            raise ValueError('解除時の指速度が表面移動上限の範囲外です')
        if not math.isfinite(finger_press_speed_m_s) or not 0<finger_press_speed_m_s<=joint_velocity_cap:
            raise ValueError("押圧時の指速度が表面移動上限の範囲外です")
        self.finger_press_speed_m_s=finger_press_speed_m_s
        self.joint_velocity_cap=joint_velocity_cap
        self.bottle_input=bottle_input
        self.finger_release_speed_m_s=finger_release_speed_m_s
        self.rest_gap=rest_gap
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -GRAVITY_M_S2, physicsClientId=self.client)
        p.setTimeStep(1 / PHYSICS_HZ, physicsClientId=self.client)
        urdf = Path(directory) / "panda-bottle.urdf"
        make_urdf(urdf)
        self.robot = p.loadURDF(str(urdf), basePosition=base,
                               baseOrientation=p.getQuaternionFromEuler((0, 0, math.pi)),
                               useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION,
                               physicsClientId=self.client)
        # Bulletではこの上限が多関節体全体へ適用される。指の低い指令速度とは分ける。
        p.changeDynamics(self.robot,0,maxJointVelocity=self.joint_velocity_cap,physicsClientId=self.client)
        names={p.getJointInfo(self.robot,i,physicsClientId=self.client)[12].decode():i
               for i in range(p.getNumJoints(self.robot,physicsClientId=self.client))}
        self.tool = names['eyedrop_bottle']
        self.fingers=[names['panda_leftfinger'],names['panda_rightfinger']]
        self.grasp=names['panda_grasptarget']
        self.obstacles = []
        self.time = 0.
        self.bottle=Bottle()
        self.drops=[]
        self.blink_padding = 0.
        self.mesh_padding = {}
        info={i:p.getJointInfo(self.robot,i,physicsClientId=self.client)
              for i in range(p.getNumJoints(self.robot,physicsClientId=self.client))}
        def cluster(link):
            while link>=0 and info[link][2]==p.JOINT_FIXED:link=info[link][16]
            return link
        adjacent={tuple(sorted((cluster(i),cluster(value[16]))))
                  for i,value in info.items() if value[2]!=p.JOINT_FIXED}
        self.allowed=set()
        for a in range(-1,len(info)):
            for b in range(a+1,len(info)):
                pair=tuple(sorted((cluster(a),cluster(b))))
                if pair[0]==pair[1] or pair in adjacent:self.allowed.add((a,b))
        for a, b in self.allowed:
            p.setCollisionFilterPair(self.robot, self.robot, a, b, 0, physicsClientId=self.client)
        self.reset(HOME)
        self.joint_info=info
        origins={j.find('child').get('link'):j.find('origin') for j in ET.parse(urdf).findall('joint')}
        self.origin_lengths={i:float(np.linalg.norm([float(v) for v in origins[value[12].decode()].get('xyz','0 0 0').split()]))
                             for i,value in info.items()}
        self.radii={}
        for shape in p.getCollisionShapeData(self.robot,-1,physicsClientId=self.client):
            if shape[2] not in (p.GEOM_MESH,p.GEOM_BOX):raise ValueError('基底の形状が未登録です')
        for link in range(-1,len(info)):
            if not p.getCollisionShapeData(self.robot,link,physicsClientId=self.client):continue
            origin=p.getBasePositionAndOrientation(self.robot,physicsClientId=self.client)[0] if link==-1 else p.getLinkState(self.robot,link,computeForwardKinematics=True,physicsClientId=self.client)[4]
            lo,hi=p.getAABB(self.robot,link,physicsClientId=self.client)
            radius=float(np.linalg.norm(np.maximum(np.abs(np.array(lo)-origin),np.abs(np.array(hi)-origin))))
            if not math.isfinite(radius) or radius>2:raise ValueError('衝突形状の包絡を検証できません')
            self.radii[link]=radius

    def reset(self, joints, *, gap=None):
        initial_gap=self.rest_gap if gap is None else gap
        if not math.isfinite(initial_gap) or not MIN_GAP_M<=initial_gap<=MAX_GAP_M:
            raise ValueError('初期指幅が物理契約外です')
        for i, q in enumerate(joints):
            p.resetJointState(self.robot, i, q, physicsClientId=self.client)
        for i in self.fingers:
            p.resetJointState(self.robot, i, initial_gap / 2, physicsClientId=self.client)
        self.time = 0.
        self.bottle=Bottle()
        self.drops=[]

    def add_geometry(self, geometry):
        frames = geometry["person_frames"]
        for index, mesh in enumerate(frames[0]):
            vertices = np.asarray(mesh["vertices"]).reshape(-1,3)
            motion=np.zeros(len(vertices))
            for frame in frames[1:]:
                other = np.asarray(frame[index]["vertices"]).reshape(-1,3)
                if other.shape != vertices.shape:
                    raise ValueError("まばたき形状の対応が変化しました")
                motion=np.maximum(motion,np.linalg.norm(other-vertices,axis=1))
            padding=float(motion.max())
            label="person:" + mesh["name"]
            self.mesh_padding[label]=padding
            self.blink_padding=max(self.blink_padding,padding)
            self._mesh(vertices.tolist(), mesh["indices"], label,motion)
        for mesh in geometry["chair"]:
            self._mesh(np.asarray(mesh["vertices"]).reshape(-1,3).tolist(), mesh["indices"], "chair:" + mesh["name"])

    def _mesh(self, vertices, indices, label,motion=None):
        vertices=np.asarray(vertices)
        # Bulletの頂点数制限に収める。三角形は間引かず、全ての面を分割登録する。
        triangles=np.asarray(indices).reshape(-1,3)
        bins=np.ceil(np.max(motion[triangles],axis=1)/BLINK_PADDING_BIN_M).astype(int) if motion is not None else np.zeros(len(triangles),dtype=int)
        for group in np.unique(bins):
            selected=triangles[bins==group].reshape(-1)
            for start in range(0,len(selected),MESH_CHUNK_INDICES):
                original,remap=np.unique(selected[start:start+MESH_CHUNK_INDICES],return_inverse=True)
                shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices[original].tolist(), indices=remap.tolist(),
                                              flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.client)
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, physicsClientId=self.client)
                self.obstacles.append((body,label,float(group)*BLINK_PADDING_BIN_M))

    def add_affine_geometry(self, geometry, blink):
        from .blink_envelope import validated_anchor_meshes
        meshes=validated_anchor_meshes(blink)
        for vertices,indices,label,padding in meshes:
            self._mesh(vertices,indices,label,padding)
        for mesh in geometry['chair']:
            self._mesh(np.asarray(mesh['vertices']).reshape(-1,3),mesh['indices'],'chair:'+mesh['name'])

    def path_bounds(self,link):
        radius=self.radii[link]
        bounds={}
        while link>=0:
            info=self.joint_info[link]
            if info[2]==p.JOINT_REVOLUTE:bounds[link]=radius*min(ARM_SPEED_RAD_S,self.joint_velocity_cap)/PHYSICS_HZ
            elif info[2]==p.JOINT_PRISMATIC:
                bounds[link]=self.joint_velocity_cap/PHYSICS_HZ
                radius+=max(abs(info[8]),abs(info[9]))
            elif info[2]!=p.JOINT_FIXED:raise ValueError('関節の移動上限が未登録です')
            radius+=self.origin_lengths[link]
            link=info[16]
        return bounds

    def own_state(self):
        joints = p.getJointStates(self.robot, list(range(7)) + self.fingers, physicsClientId=self.client)
        return [j[0] for j in joints[:7]] + [joints[7][0]+joints[8][0]] + [j[1] for j in joints[:7]] + [joints[7][1]+joints[8][1]]

    def ik(self, nozzle):
        target = [nozzle[0], nozzle[1], nozzle[2] + NOZZLE_GRASP_OFFSET_M]
        return p.calculateInverseKinematics(self.robot, self.grasp, target,
                    targetOrientation=p.getQuaternionFromEuler((math.pi,0,math.pi)),
                    maxNumIterations=1000, residualThreshold=1e-7, physicsClientId=self.client)[:7]

    def nozzle(self):
        state = p.getLinkState(self.robot,self.tool,computeForwardKinematics=True,physicsClientId=self.client)
        return p.multiplyTransforms(state[4],state[5],NOZZLE_LOCAL,(0,0,0,1))[0]

    def nozzle_kinematics(self):
        state=p.getLinkState(self.robot,self.tool,computeForwardKinematics=True,computeLinkVelocity=True,physicsClientId=self.client)
        position=p.multiplyTransforms(state[4],state[5],NOZZLE_LOCAL,(0,0,0,1))[0]
        axis=np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3,3)[:,2]
        return nozzle_motion(position,state[0],state[6],state[7],axis)

    def contacts(self, swept=False):
        contacts=[]
        bounds={link:self.path_bounds(link) if swept else {} for link in self.radii}
        travel={link:sum(value.values()) for link,value in bounds.items()}
        maximum=max(travel.values())
        for body,label,padding in self.obstacles:
            search=CONTACT_MARGIN_M+maximum+padding
            for point in p.getClosestPoints(self.robot,body,search,physicsClientId=self.client):
                margin=CONTACT_MARGIN_M+travel[point[3]]+padding
                if point[8]<=margin:contacts.append({"kind":label,"link":point[3],"distance":point[8],"margin":margin})
        for point in p.getClosestPoints(self.robot,self.robot,CONTACT_MARGIN_M+2*maximum,physicsClientId=self.client):
            a,b=sorted((point[3],point[4]))
            if a!=b and (a,b) not in self.allowed:
                exclusive=bounds[a].keys()^bounds[b].keys()
                margin=CONTACT_MARGIN_M+sum(bounds[a].get(i,0)+bounds[b].get(i,0) for i in exclusive)
                if point[8]<=margin:contacts.append({"kind":"self","links":[a,b],"distance":point[8],"margin":margin})
        nearest={}
        for contact in contacts:
            key=(contact['kind'],contact.get('link'),tuple(contact.get('links',[])))
            if key not in nearest or contact['distance']<nearest[key]['distance']:nearest[key]=contact
        return list(nearest.values())

    def step(self, action, *, audit_pressure=False, audit_motion=False):
        if audit_motion and self.drop_model!=MOVING_NOZZLE:
            raise ValueError('運動区間の監査にはmoving-nozzle-v2を明示してください')
        action=np.asarray(action,dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all() or not MIN_GAP_M<=action[7]<=MAX_GAP_M:
            raise ValueError("動作が契約外です")
        for i in range(7):
            info=p.getJointInfo(self.robot,i,physicsClientId=self.client)
            if not info[8]<=action[i]<=info[9]:raise ValueError("関節指令が可動域外です")
            p.setJointMotorControl2(self.robot,i,p.POSITION_CONTROL,targetPosition=action[i],
                                   force=info[10],maxVelocity=min(ARM_SPEED_RAD_S,self.joint_velocity_cap),physicsClientId=self.client)
        closing_speed=self.finger_press_speed_m_s if action[7]<BOTTLE_DEPTH_M else FINGER_SPEED_M_S
        release_speed=self.finger_release_speed_m_s if action[7]>self.own_state()[7] else closing_speed
        for i in self.fingers:
            p.setJointMotorControl2(self.robot,i,p.POSITION_CONTROL,targetPosition=action[7]/2,
                                   force=20,maxVelocity=release_speed,physicsClientId=self.client)
        segments=[]
        domain_samples=[]
        motion_samples=[]
        moving=self.drop_model==MOVING_NOZZLE
        pressure_samples=[{'time':self.time,'gap_m':self.own_state()[7]}] if audit_pressure else None
        for _ in range(SUBSTEPS):
            contact=self.contacts(swept=True)
            if contact:
                result={"contacts":contact,"time":self.time,"segments":segments}
                if audit_pressure:result['pressure_samples']=pressure_samples
                if moving:result['drop_domain_samples']=domain_samples
                if audit_motion:result['motion_samples']=motion_samples
                return result
            if moving:
                interval_start=self.time;motion_before=self.nozzle_kinematics();bottle_before=self.bottle.snapshot()
                if audit_motion:own_before=self.own_state()
            p.stepSimulation(physicsClientId=self.client)
            self.time+=1/PHYSICS_HZ
            if audit_pressure:pressure_samples.append({'time':self.time,'gap_m':self.own_state()[7]})
            displacement=(BOTTLE_DEPTH_M-self.own_state()[7])*1000
            if displacement>self.bottle.config.max_displacement_mm+GAP_TOLERANCE_M*1000:
                raise RuntimeError('実測押圧が容器の上限を超えました')
            update_bottle=self.bottle.step_measured if self.bottle_input=='measured' else self.bottle.step
            events=update_bottle(max(0.,min(displacement,self.bottle.config.max_displacement_mm)),1/PHYSICS_HZ)
            if moving:
                motion_after=self.nozzle_kinematics()
                bottle_after=self.bottle.snapshot()
                domain_samples.append(domain_sample(interval_start,self.time,motion_before,motion_after,
                    bottle_before,bottle_after,len(events)))
                emissions=[]
                for event in events:
                    self.drops.append(emit_drop(len(self.drops),event.time_s,interval_start,self.time,motion_before,motion_after))
                    if audit_motion:
                        drop=self.drops[-1]
                        emissions.append(dict(id=drop['id'],emitted_at=drop['emitted_at'],
                            position=list(drop['position']),velocity=list(drop['velocity'])))
                if audit_motion:
                    motion_samples.append(dict(start_time=interval_start,time=self.time,before=motion_before,after=motion_after,
                        own_before=own_before,own_after=self.own_state(),bottle_before=bottle_before,bottle_after=bottle_after,
                        emissions=emissions))
                for drop in self.drops:
                    segment=advance_drop(drop,interval_start,self.time)
                    if segment is not None:segments.append(dict(segment,start=to_scene(segment['start']),end=to_scene(segment['end'])))
            else:
                # 旧結果の再現用。新しい全動作比較は移動ノズルのモデルを明示する。
                for event in events:
                    self.drops.append({'id':len(self.drops),'position':list(self.nozzle()),'velocity':[0.,0.,0.],'emitted_at':self.time})
                for drop in self.drops:
                    old=drop['position'][:]
                    drop['position']=[old[i]+drop['velocity'][i]/PHYSICS_HZ-(GRAVITY_M_S2/(2*PHYSICS_HZ**2) if i==2 else 0) for i in range(3)]
                    drop['velocity'][2]-=GRAVITY_M_S2/PHYSICS_HZ
                    segments.append({'id':drop['id'],'time':self.time,'start':to_scene(old),'end':to_scene(drop['position'])})
            velocity=self.own_state()[8:]
            if max(abs(v) for v in velocity[:7])>self.joint_velocity_cap+1e-9:
                raise RuntimeError(f"表面移動上限を超える関節速度: {velocity[:7]}")
            finger_states=p.getJointStates(self.robot,self.fingers,physicsClientId=self.client)
            if max(abs(value[1]) for value in finger_states)>self.joint_velocity_cap:
                raise RuntimeError(f'表面移動上限を超える指速度: {[value[1] for value in finger_states]}')
        result={"contacts":self.contacts(swept=True),"time":self.time,"segments":segments}
        if audit_pressure:result['pressure_samples']=pressure_samples
        if moving:result['drop_domain_samples']=domain_samples
        if audit_motion:result['motion_samples']=motion_samples
        return result

    def drawing(self):
        assets=Path(pybullet_data.getDataPath())/"franka_panda"
        shapes=[]
        for shape in p.getVisualShapeData(self.robot,physicsClientId=self.client):
            kind={p.GEOM_MESH:"mesh",p.GEOM_BOX:"box",p.GEOM_SPHERE:"sphere"}.get(shape[2])
            if kind is None:raise ValueError("描画の形状が未登録です")
            filename=shape[4].decode()
            shapes.append({"link":shape[1],"kind":kind,"dimensions":shape[3],"mesh":str(Path(filename).relative_to(assets)) if filename else "",
                           "position":shape[5],"orientation":shape[6],"color":shape[7]})
        return shapes

    def links(self):
        links=[]
        visible={shape[1] for shape in p.getVisualShapeData(self.robot,physicsClientId=self.client)}
        # ベースは慣性原点からURDF原点に戻す。
        position,rotation=p.getBasePositionAndOrientation(self.robot,physicsClientId=self.client)
        dynamics=p.getDynamicsInfo(self.robot,-1,physicsClientId=self.client)
        inv=p.invertTransform(dynamics[3],dynamics[4]);position,rotation=p.multiplyTransforms(position,rotation,*inv)
        links.append({"link":-1,"position":position,"orientation":rotation})
        for i in range(p.getNumJoints(self.robot,physicsClientId=self.client)):
            if i not in visible:continue
            value=p.getLinkState(self.robot,i,computeForwardKinematics=True,physicsClientId=self.client)
            links.append({"link":i,"position":value[4],"orientation":value[5]})
        return links

    def close(self):
        p.disconnect(self.client)
