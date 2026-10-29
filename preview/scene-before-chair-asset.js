import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin} from './node_modules/@pixiv/three-vrm/lib/three-vrm.module.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const C={skin:0xc99172,lip:0x975950,hair:0x302d28,shirt:0x547c78,pants:0x344553,
 chair:0x283e42,metal:0x869895,sole:0x243232,floor:0xe8eee7};
const N={segments:48,rings:32,eyeRadius:.018,lidRadius:.0192,close:.095,hold:.04,open:.17,
 blinkStarts:[1.1,3.65,4.12,7.3],cycle:9,torsoTilt:-.35,headTilt:-.48};
const ARM_POSE={wristX:.27,wristY:.818,wristZ:.17,poleX:.35,poleY:-1,poleZ:-.4,
 fingerCurl:{Proximal:.10,Intermediate:.18,Distal:.12}};
const scene=new THREE.Scene();scene.background=new THREE.Color(C.floor);
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
document.body.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(34,innerWidth/innerHeight,.005,30);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const mat=(color,roughness=.65)=>new THREE.MeshStandardMaterial({color,roughness});
const skin=mat(C.skin),shirt=mat(C.shirt),pants=mat(C.pants),hair=mat(C.hair),chair=mat(C.chair);
function mesh(geo,material,parent=scene){const m=new THREE.Mesh(geo,material);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function ellipsoid(pos,scale,material,parent=scene){
 const m=mesh(new THREE.SphereGeometry(1,N.segments,N.rings),material,parent);m.position.set(...pos);m.scale.set(...scale);return m;
}
function box(pos,size,material,parent=scene){
 const m=mesh(new THREE.BoxGeometry(...size),material,parent);m.position.set(...pos);
 return m;
}
function bar(a,b,r,material,parent=scene){
 const va=new THREE.Vector3(...a),vb=new THREE.Vector3(...b),d=vb.clone().sub(va);
 const m=mesh(new THREE.CylinderGeometry(r,r,d.length(),24),material,parent);
 m.position.copy(va.add(vb).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());return m;
}
function line(points,r,material,parent){
 const curve=new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p)));
 return mesh(new THREE.TubeGeometry(curve,40,r,8,false),material,parent);
}
scene.add(new THREE.HemisphereLight(0xffffff,0x7e9890,2.1));
function light(pos,power,size){const l=new THREE.DirectionalLight(0xfff4e6,power);l.position.set(...pos);l.castShadow=true;l.shadow.mapSize.set(2048,2048);l.shadow.camera.left=-2;l.shadow.camera.right=2;l.shadow.camera.top=2;l.shadow.camera.bottom=-2;l.shadow.bias=-.0003;scene.add(l);}
light([2,4,3],3);scene.add(new THREE.AmbientLight(0xffffff,.25));
const floor=mesh(new THREE.PlaneGeometry(20,20),mat(C.floor));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;

// 五本脚・キャスター・昇降支柱・背もたれ・ヘッドレスト。
const metal=mat(C.metal,.35);
bar([0,.09,0],[0,.49,0],.035,metal);
for(let i=0;i<5;i++){const a=i*Math.PI*2/5;const end=[Math.sin(a)*.34,.07,Math.cos(a)*.34];
 bar([0,.13,0],end,.022,metal);ellipsoid([end[0],.04,end[2]],[.033,.04,.042],chair);}
ellipsoid([0,.51,.02],[.285,.065,.26],chair);
const back=new THREE.Group();scene.add(back);back.position.set(0,.54,-.15);back.rotation.x=N.torsoTilt;
ellipsoid([0,.29,-.05],[.255,.32,.055],chair,back);
ellipsoid([0,.69,-.055],[.15,.10,.055],chair,back);
for(let i=-1;i<=1;i+=2){
 bar([i*.28,.49,0],[i*.28,.76,0],.018,metal);
 ellipsoid([i*.28,.77,.08],[.055,.028,.18],chair);
}

// VRMの正規化骨格で着座と顔の上向きを設定する。
const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));
const gltf=await loader.loadAsync('./assets/avatar.vrm');
const vrm=gltf.userData.vrm;scene.add(vrm.scene);
const bone=name=>vrm.humanoid.getNormalizedBoneNode(name);
bone('hips').rotation.x=-.30;
for(const side of ['left','right']){
 const sign=side==='left'?1:-1;
 bone(side+'UpperLeg').rotation.x=-Math.PI/2+.30;
 bone(side+'LowerLeg').rotation.x=Math.PI/2;
 bone(side+'UpperArm').rotation.z=-sign*1.15;
 bone(side+'LowerArm').rotation.y=-sign*1.2;
}
bone('neck').rotation.x=-.20;bone('head').rotation.x=-.20;
vrm.update(0);scene.updateMatrixWorld(true);
const hipPosition=bone('hips').getWorldPosition(new THREE.Vector3());
vrm.scene.position.y+=.61-hipPosition.y;
vrm.update(0);scene.updateMatrixWorld(true);
// 肘掛け上の手首を目標に、上腕と前腕の長さを保って二関節を配置する。
const worldPosition=node=>node.getWorldPosition(new THREE.Vector3());
function aimBone(node,child,target){
 const origin=worldPosition(node);
 const current=worldPosition(child).sub(origin).normalize();
 const desired=target.clone().sub(origin).normalize();
 const turn=new THREE.Quaternion().setFromUnitVectors(current,desired);
 const world=turn.multiply(node.getWorldQuaternion(new THREE.Quaternion()));
 node.quaternion.copy(node.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(world));
 node.updateWorldMatrix(false,true);
}
for(const side of ['left','right']){
 const sign=side==='left'?1:-1;
 const upper=bone(side+'UpperArm'),lower=bone(side+'LowerArm'),hand=bone(side+'Hand');
 const shoulder=worldPosition(upper),elbow=worldPosition(lower),wrist=worldPosition(hand);
 const target=new THREE.Vector3(sign*ARM_POSE.wristX,ARM_POSE.wristY,ARM_POSE.wristZ);
 const upperLength=elbow.distanceTo(shoulder),lowerLength=wrist.distanceTo(elbow);
 const direction=target.clone().sub(shoulder),distance=direction.length();direction.normalize();
 if(distance>=upperLength+lowerLength||distance<=Math.abs(upperLength-lowerLength))throw new Error('手首の目標が腕の可動範囲外です');
 const along=(upperLength**2-lowerLength**2+distance**2)/(2*distance);
 const height=Math.sqrt(upperLength**2-along**2);
 const pole=new THREE.Vector3(sign*ARM_POSE.poleX,ARM_POSE.poleY,ARM_POSE.poleZ);
 pole.addScaledVector(direction,-pole.dot(direction)).normalize();
 const targetElbow=shoulder.clone().addScaledVector(direction,along).addScaledVector(pole,height);
 aimBone(upper,lower,targetElbow);aimBone(lower,hand,target);
 // 掌を下、指先を前に向ける。前腕の回転を手首へ持ち越さない。
 const palmDown=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),-sign*Math.PI/2);
 hand.quaternion.copy(hand.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(palmDown));
 for(const finger of ['Index','Middle','Ring','Little'])for(const [joint,angle] of Object.entries(ARM_POSE.fingerCurl)){
  const segment=bone(side+finger+joint);if(segment)segment.rotation.z=-sign*angle;
 }
}
vrm.update(0);scene.updateMatrixWorld(true);
vrm.scene.traverse(obj=>{if(obj.isMesh){obj.castShadow=true;obj.receiveShadow=false;obj.frustumCulled=false;}});
const head=bone('head');
let openness=1,last=-1,paused=false,manual=false,clock=0,view='wide';
function setOpen(value){openness=value;vrm.expressionManager.setValue('blink',1-value);vrm.update(0);last=value;}
export function blink(t){
 const phase=t%N.cycle;
 for(const start of N.blinkStarts){let q=phase-start;
  if(q<0)continue;if(q<N.close)return (1+Math.cos(Math.PI*q/N.close))/2;
  q-=N.close;if(q<N.hold)return 0;q-=N.hold;
  if(q<N.open)return (1-Math.cos(Math.PI*q/N.open))/2;
 }
 return 1;
}
function setView(name){
 view=name;scene.updateMatrixWorld(true);
 if(name==='face'){
  const p=head.getWorldPosition(new THREE.Vector3()),q=head.getWorldQuaternion(new THREE.Quaternion());
  controls.target.copy(new THREE.Vector3(0,.07,0).applyQuaternion(q).add(p));
  camera.position.copy(new THREE.Vector3(.05,.07,.68).applyQuaternion(q).add(p));
 }else if(name==='side'){controls.target.set(0,.83,.04);camera.position.set(2.7,1.2,.45);
 }else{controls.target.set(0,.79,.02);camera.position.set(2.15,1.75,2.65);}
 controls.update();
}
document.getElementById('wide').onclick=()=>setView('wide');
document.getElementById('face').onclick=()=>setView('face');
document.getElementById('side').onclick=()=>setView('side');
document.getElementById('pause').onclick=()=>{paused=!paused;document.getElementById('pause').textContent=paused?'再生':'一時停止';};
document.getElementById('open').oninput=e=>{manual=true;setOpen(Number(e.target.value));};
document.getElementById('auto').onclick=()=>{manual=false;paused=false;};
function renderAt(t,value=null){
 clock=t;if(value!==null){manual=true;setOpen(value);}else if(!manual)setOpen(blink(t));
 document.getElementById('status').textContent=`${t.toFixed(2)}秒 ｜ 開眼量 ${(openness*100).toFixed(0)}%`;
 document.getElementById('open').value=openness;renderer.render(scene,camera);
}
window.preview={setView,renderAt,seek:(t)=>{manual=false;paused=true;renderAt(t);},pause:()=>{paused=true;},info:()=>({openness,view,expressions:Object.keys(vrm.expressionManager.expressionMap),head:head.getWorldPosition(new THREE.Vector3()).toArray(),meshCount:scene.children.length})};
setView('wide');let prev=performance.now();
renderer.setAnimationLoop(now=>{const dt=Math.min((now-prev)/1000,.1);prev=now;if(!paused)clock+=dt;controls.update();renderAt(clock);});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
window.previewReady=true;
