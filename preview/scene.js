import {learningConditions} from './learning-conditions.mjs';
import {createDiagnostic} from './diagnostic.js';
import {irisCenter} from './eye-target.js';
import {firstSurfaceHit} from './anatomy-contact.mjs';
import {classifySurface,SCORING_VERSION} from './anatomy-score.mjs';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from './node_modules/@pixiv/three-vrm/lib/three-vrm.module.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const C={skin:0xc99172,lip:0x975950,hair:0x302d28,shirt:0x547c78,pants:0x344553,
 chair:0x283e42,metal:0x869895,sole:0x243232,floor:0xe8eee7};
const N={segments:48,rings:32,eyeRadius:.018,lidRadius:.0192,close:.095,hold:.04,open:.17,
 blinkStarts:[1.1,3.65,4.12,7.3],cycle:9,torsoTilt:-.35,headTilt:-.48};
const ARM_POSE={wristX:.27,wristY:.818,wristZ:.17,poleX:.35,poleY:-1,poleZ:-.4,
 fingerCurl:{Proximal:.10,Intermediate:.18,Distal:.12}};
const HEAD_POSE={neck:-.35,head:-.35};
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

// CC0の椅子素材を着座姿勢に合わせ、背もたれだけを倒す。
const CHAIR_POSE={widthScale:.875,backPivot:[0,.54,-.15],tilt:-.35,
 headrest:[0,1.23,-.205],headrestSize:[.15,.10,.055],armForward:.10};
const chairAsset=await new GLTFLoader().loadAsync('./assets/office-chair.glb');
const chairRoot=chairAsset.scene;scene.add(chairRoot);chairRoot.scale.x=CHAIR_POSE.widthScale;
const back=new THREE.Group();chairRoot.add(back);back.position.set(...CHAIR_POSE.backPivot);
chairRoot.updateMatrixWorld(true);
const backParts=[];
chairRoot.traverse(obj=>{if(obj.isMesh){obj.castShadow=true;obj.receiveShadow=true;}
 if(obj.name.startsWith('Back'))backParts.push(obj);
 if(obj.name.startsWith('Arm'))obj.position.z+=CHAIR_POSE.armForward;
});
for(const part of backParts)back.attach(part);
// 素材にないヘッドレストと支持棒は追加部品。
const headrest=ellipsoid(CHAIR_POSE.headrest,CHAIR_POSE.headrestSize,chair,chairRoot);
const support=bar([0,1.08,-.25],[0,1.23,-.25],.018,mat(C.metal,.35),chairRoot);
chairRoot.updateMatrixWorld(true);back.attach(headrest);back.attach(support);
back.rotation.x=CHAIR_POSE.tilt;

// VRMの正規化骨格で着座と顔の上向きを設定する。
const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));
const peopleManifest=await fetch('./assets/people/manifest.json').then(r=>r.json());
const people=[{id:'original',credit:'Model © 2022 pixiv Inc. · VRM Public License 1.0'},
 ...peopleManifest.models.filter(m=>m.status==='downloaded_structurally_checked')];
const person=new URLSearchParams(location.search).get('person')||'original';
const selected=people.find(p=>p.id===person);
if(!selected)throw new Error('指定された人物モデルはありません');
const chooser=document.createElement('select');chooser.setAttribute('aria-label','人物モデル');
for(const item of people)chooser.add(new Option(item.id,item.id));
chooser.value=person;chooser.onchange=()=>{const url=new URL(location.href);url.searchParams.set('person',chooser.value);location.href=url;};
document.querySelector('aside').prepend(chooser);
document.querySelector('footer').textContent=`${selected.credit||selected.author} ｜ 画像ACTと正解位置の診断を別々に実行`;
const gltf=await loader.loadAsync(person==='original'?'./assets/avatar.vrm':`./assets/people/${person}.vrm`);
const vrm=gltf.userData.vrm;VRMUtils.rotateVRM0(vrm);scene.add(vrm.scene);
const annotations=await fetch('./assets/anatomy-annotations.json').then(r=>r.json());
const anatomyAnnotation=annotations.models[person]||null;
if(anatomyAnnotation){
 const url=person==='original'?'./assets/avatar.vrm':`./assets/people/${person}.vrm`;
 const bytes=await fetch(url).then(r=>r.arrayBuffer());
 const digest=await crypto.subtle.digest('SHA-256',bytes);
 const hash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
 if(hash!==anatomyAnnotation.asset_sha256)throw new Error('人物素材と採点注釈の版が一致しません');
}
// 旧VRMは正面補正で骨格の前後軸も反転するため、曲げ方向を合わせる。
const poseSign=vrm.meta.metaVersion==='0'?-1:1;
const bone=name=>vrm.humanoid.getNormalizedBoneNode(name);
bone('hips').rotation.x=-.30*poseSign;
for(const side of ['left','right']){
 const sign=side==='left'?1:-1;
 bone(side+'UpperLeg').rotation.x=(-Math.PI/2+.30)*poseSign;
 bone(side+'LowerLeg').rotation.x=Math.PI/2*poseSign;
 bone(side+'UpperArm').rotation.z=-sign*1.15;
 bone(side+'LowerArm').rotation.y=-sign*1.2;
}
bone('neck').rotation.x=HEAD_POSE.neck*poseSign;bone('head').rotation.x=HEAD_POSE.head*poseSign;
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
 const wristZ=selected.seatedWristZ??ARM_POSE.wristZ;
 const target=new THREE.Vector3(sign*ARM_POSE.wristX,ARM_POSE.wristY,wristZ);
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
 const palmDown=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),-sign*Math.PI/2*poseSign);
 hand.quaternion.copy(hand.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(palmDown));
 for(const finger of ['Index','Middle','Ring','Little'])for(const [joint,angle] of Object.entries(ARM_POSE.fingerCurl)){
  const segment=bone(side+finger+joint);if(segment)segment.rotation.z=-sign*angle*poseSign;
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
  camera.position.copy(new THREE.Vector3(.05,.07,.68*poseSign).applyQuaternion(q).add(p));
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
const eyeTargetAvailable=Boolean(anatomyAnnotation?.eye_surfaces?.length);
window.preview={setView,renderAt,seek:(t)=>{manual=false;paused=true;renderAt(t);},pause:()=>{paused=true;},info:()=>({person,eyeTargetAvailable,openness,view,expressions:Object.keys(vrm.expressionManager.expressionMap),head:head.getWorldPosition(new THREE.Vector3()).toArray(),meshCount:scene.children.length})};
// 採点・教師専用。登録済みの虹彩面から位置を求め、推論へは渡さない。
const EYE_SURFACE_OFFSET=.001;
function eyeSurface(){
 const previousOpen=openness;
 setOpen(1);scene.updateMatrixWorld(true);
 const meshes=[];
 vrm.scene.traverse(obj=>{
  if(!obj.isMesh)return;
  meshes.push(obj);
 });
 try{
  if(!anatomyAnnotation?.eye_surfaces?.length)throw new Error('採点用の虹彩注釈がありません');
  const center=irisCenter(meshes,head,anatomyAnnotation.eye_surfaces);
  const forward=new THREE.Vector3(0,0,EYE_SURFACE_OFFSET*poseSign).applyQuaternion(head.getWorldQuaternion(new THREE.Quaternion()));
  return center.add(forward);
 }finally{setOpen(previousOpen);scene.updateMatrixWorld(true);}
}
const diagnostic=createDiagnostic({scene,eye:eyeSurface,person:vrm.scene,
 apply:state=>{paused=true;renderAt(state.time,state.openness);}});
window.preview.diagnostic=diagnostic;
window.preview.eyeTarget=()=>eyeSurface().toArray();
// 学習カメラは眼の正解位置を参照せず、全人物で同じ外部配置に固定する。
const learningCamera=new THREE.PerspectiveCamera(40,1,.01,10);
learningCamera.position.set(.12,1.55,.6);learningCamera.lookAt(0,1.20,-.1);
const learningRenderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
learningRenderer.setSize(256,256);
learningRenderer.toneMapping=renderer.toneMapping;learningRenderer.toneMappingExposure=renderer.toneMappingExposure;
const basePersonPosition=vrm.scene.position.clone();
let learningPhase=0;
const learningScenario=new URLSearchParams(location.search).get("scenario")||"v6";
window.learningScene={
 scenario:learningScenario,
 reset(seed){
  paused=true;manual=true;learningPhase=(seed%7)*.27;
  const condition=learningConditions(seed,learningScenario);
  vrm.scene.position.copy(basePersonPosition).add(new THREE.Vector3(...condition.offset));
  bone('head').rotation.set((HEAD_POSE.head+condition.pitch)*poseSign,condition.yaw,0);
  vrm.update(0);scene.updateMatrixWorld(true);
  return condition;
 },
 render(state){
  const open=blink(state.time+learningPhase);
  const contact=diagnostic.renderLearning(state,open);
  learningRenderer.render(scene,learningCamera);
  return {contact};
 },
 capture(){return learningRenderer.domElement.toDataURL('image/png');},
 surfaceHit(start,end){
  return firstSurfaceHit(vrm.scene,new THREE.Vector3(...start),new THREE.Vector3(...end),
   anatomyAnnotation?.visual_surfaces||[]);
 },
 scoring:{version:SCORING_VERSION,annotation_version:annotations.version,
  available:anatomyAnnotation!==null,asset_sha256:anatomyAnnotation?.asset_sha256||null},
 classifyHit(hit,open){return classifySurface(hit,open,anatomyAnnotation);},
 teacher(){return {eye:eyeTargetAvailable?eyeSurface().toArray():null,openness};}
};
const {mountLearningRunner}=await import('./learned-run.js');
mountLearningRunner(window.learningScene);
window.preview.handPoints=()=>Object.fromEntries(['left','right'].map(side=>[side,
 Object.fromEntries(['Hand','IndexProximal','MiddleProximal','LittleProximal','MiddleDistal'].map(name=>[
 name,vrm.humanoid.getRawBoneNode(side+name)?.getWorldPosition(new THREE.Vector3()).toArray()]))]));
if(!eyeTargetAvailable){
 document.getElementById('diagnostic').disabled=true;
 document.getElementById('robot-status').textContent='正解座標を使う診断の基準点が未設定です。画像からの学習ができないという意味ではありません。';
}
if(new URLSearchParams(location.search).get('inspect')==='anatomy'){
 // 独立採点の形状検査専用。方策と教師の操作には接続しない。
 window.anatomyInspector={
  probe:(start,end)=>firstSurfaceHit(vrm.scene,new THREE.Vector3(...start),new THREE.Vector3(...end)),
  pick:(x,y)=>{
   const ray=new THREE.Raycaster();
   ray.setFromCamera(new THREE.Vector2(x,y),camera);
   const annotationRayLength=10;
   return firstSurfaceHit(vrm.scene,ray.ray.origin,
    ray.ray.at(annotationRayLength,new THREE.Vector3()));
  },
  materials:()=>{
   const result=[];
   vrm.scene.traverse(mesh=>{if(mesh.isMesh)result.push({mesh:mesh.name,material:mesh.material.name,
    triangles:(mesh.geometry.index?.count||mesh.geometry.attributes.position.count)/3});});
   return result;
  }
 };
}
setView('wide');let prev=performance.now();
renderer.setAnimationLoop(now=>{const dt=Math.min((now-prev)/1000,.1);prev=now;if(!paused)clock+=dt;controls.update();renderAt(clock);});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
window.previewReady=true;
