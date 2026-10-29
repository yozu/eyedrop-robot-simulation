// scene.jsの末尾へ読み込み、旧プレビューのソースは変更しない。
import {pandaCondition} from './panda-condition.mjs';
import {bodyLeftIrisCenter} from './eye-target.js';
import {cameraVisibility} from './camera-visibility.mjs';
import {classifySpecifiedEye,SPECIFIED_EYE_SCORING_VERSION} from './anatomy-score.mjs';
const PANDA_EXPECTED_EYE='left';
renderer.setAnimationLoop(null);
paused=true;manual=true;
// 毛髪のばねを止め、まばたき以外の形状が呼出回数で変化しないようにする。
if(vrm.springBoneManager)vrm.springBoneManager.update=()=>{};
learningRenderer.setSize(512,512);
const pandaCameraRight=learningCamera.clone();
pandaCameraRight.position.add(new THREE.Vector3(1,0,0).applyQuaternion(learningCamera.quaternion).multiplyScalar(.16));
pandaCameraRight.updateMatrixWorld(true);
const pandaRoot=new THREE.Group();scene.add(pandaRoot);
const pandaLinks=new Map();
const pandaFrame=new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/2,0,0));
function physicsPoint(p){return new THREE.Vector3(p[0],p[2],-p[1]);}
function posedGeometry(root){
 const meshes=[];
 root.updateMatrixWorld(true);
 root.traverse(mesh=>{
  if(!mesh.isMesh||!mesh.geometry?.attributes?.position)return;
  mesh.skeleton?.update();
  const vertices=[],v=new THREE.Vector3(),attribute=mesh.geometry.attributes.position;
  for(let i=0;i<attribute.count;i++){
   mesh.getVertexPosition(i,v);v.applyMatrix4(mesh.matrixWorld);
   vertices.push(v.x,-v.z,v.y);
  }
  const indices=mesh.geometry.index?Array.from(mesh.geometry.index.array):Array.from({length:attribute.count},(_,i)=>i);
  meshes.push({name:mesh.name,vertices,indices});
 });
 return meshes;
}
window.pandaExperiment={
 diagnosticParts(anchors){
  // 元の顔配置教師と同じアンカーを投影する。認識・制御には接続しない。
  const names=['left_eye','right_eye','nose_tip','mouth_center'],result={};
  const weightTolerance=1e-6;
  for(const name of names){
   const anchor=anchors?.[name];
   if(!anchor){result[name]=null;continue;}
   if(!Array.isArray(anchor.vertices)||!Array.isArray(anchor.weights)||anchor.vertices.length!==anchor.weights.length||
      !anchor.vertices.length||!anchor.weights.every(Number.isFinite)||Math.abs(anchor.weights.reduce((a,b)=>a+b,0)-1)>weightTolerance)
    throw new Error('顔配置アンカーが不正です');
   const mesh=vrm.scene.getObjectByName(anchor.mesh);
   if(!mesh?.isMesh)throw new Error('顔配置教師のメッシュがありません');
   mesh.skeleton?.update();
   const point=new THREE.Vector3();
   for(let i=0;i<anchor.vertices.length;i++){
    const vertex=anchor.vertices[i];
    if(!Number.isInteger(vertex)||vertex<0||vertex>=mesh.geometry.attributes.position.count)throw new Error('顔配置教師の頂点が範囲外です');
    point.addScaledVector(mesh.getVertexPosition(vertex,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld),anchor.weights[i]);
   }
   const ndc=point.clone().project(learningCamera);
   result[name]={scene:point.toArray(),pixel:[(ndc.x+1)/2,(1-ndc.y)/2]};
  }
  return result;
 },
 visibility(point){
  if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new Error('診断用視認点が不正です');
  const target=new THREE.Vector3(...point),groups=[{root:vrm.scene,label:'person'},{root:chairRoot,label:'chair'},{root:pandaRoot,label:'robot'}];
  return {left:cameraVisibility(learningCamera,target,groups),right:cameraVisibility(pandaCameraRight,target,groups)};
 },
 blinkGeometry(){
  const previousOpen=openness,manager=vrm.expressionManager,expression=manager.getExpression('blink');
  if(!expression||typeof expression.isBinary!=='boolean'||!Array.isArray(manager.expressions))throw new Error('まばたき補間の構造を確認できません');
  const values=[1,.75,.5,.25,0],frames=[],weights=[];let firstBones=null,bonesUnchanged=true;
  try{
   for(const value of values){
    setOpen(value);scene.updateMatrixWorld(true);frames.push(posedGeometry(vrm.scene));
    const bones=[];vrm.scene.traverse(mesh=>{if(mesh.isSkinnedMesh){mesh.skeleton.update();bones.push(...mesh.skeleton.boneMatrices);}});
    if(firstBones===null)firstBones=bones;
    else if(bones.length!==firstBones.length||bones.some((v,i)=>v!==firstBones[i]))bonesUnchanged=false;
    weights.push(manager.expressions.map(e=>({name:e.expressionName,weight:e.weight,isBinary:e.isBinary,overrideBlink:e.overrideBlink})));
   }
   return {values,frames,weights,bonesUnchanged,isBinary:expression.isBinary,overrideBlink:expression.overrideBlink,
    binds:expression.binds.map(b=>b.constructor.name)};
  }finally{setOpen(previousOpen);scene.updateMatrixWorld(true);}
 },
 teacherV2(){
  const previousOpen=openness;setOpen(1);scene.updateMatrixWorld(true);
  try{
   const left=vrm.humanoid.getRawBoneNode('leftUpperArm');
   const right=vrm.humanoid.getRawBoneNode('rightUpperArm');
   if(!left||!right)throw new Error('教師用の身体左右ボーンがありません');
   const meshes=[];vrm.scene.traverse(obj=>{if(obj.isMesh)meshes.push(obj);});
   const selected=bodyLeftIrisCenter(meshes,head,anatomyAnnotation?.eye_surfaces,
    left.getWorldPosition(new THREE.Vector3()),right.getWorldPosition(new THREE.Vector3()));
   const forward=new THREE.Vector3(0,0,EYE_SURFACE_OFFSET*poseSign).applyQuaternion(head.getWorldQuaternion(new THREE.Quaternion()));
   return {version:'body-left-iris-ray-v2',legacy_proxy:eyeSurface().toArray(),
    proxy:selected.center.clone().add(forward).toArray(),localSide:selected.localSide,
    bodyLeftAxis:selected.bodyLeftAxis.toArray(),candidates:selected.candidates.map(c=>({
     localSide:c.localSide,proxy:c.center.clone().add(forward).toArray(),projection:c.projection}))};
  }finally{setOpen(previousOpen);scene.updateMatrixWorld(true);}
 },
 eyeAnatomy(){
  const positions={};
  for(const side of ["leftEye","rightEye"]){
   const node=vrm.humanoid.getRawBoneNode(side);
   positions[side]=node?node.getWorldPosition(new THREE.Vector3()).toArray():null;
  }
  const body={};
  for(const name of ["head","leftUpperArm","rightUpperArm"]){
   const node=vrm.humanoid.getRawBoneNode(name);body[name]=node?node.getWorldPosition(new THREE.Vector3()).toArray():null;
  }
  return {proxy:eyeSurface().toArray(),rawEyes:positions,rawBody:body};
 },
 reset(depth=0,phase=0,height=0,horizontal=0){
  this.scoringEyes=null;
  const condition=pandaCondition({depth,phase,height,horizontal});
  vrm.scene.position.copy(basePersonPosition).add(new THREE.Vector3(...condition.offset));
  bone('head').rotation.set(HEAD_POSE.head*poseSign,0,0);
  learningPhase=phase;this.scored=new Set();setOpen(1);vrm.update(0);scene.updateMatrixWorld(true);
  const left=vrm.humanoid.getRawBoneNode('leftUpperArm');
  const right=vrm.humanoid.getRawBoneNode('rightUpperArm');
  if(!left||!right)throw new Error('採点用の身体左右ボーンがありません');
  const meshes=[];vrm.scene.traverse(obj=>{if(obj.isMesh)meshes.push(obj);});
  const selected=bodyLeftIrisCenter(meshes,head,anatomyAnnotation?.eye_surfaces,
   left.getWorldPosition(new THREE.Vector3()),right.getWorldPosition(new THREE.Vector3()));
  this.scoringEyes={left:selected.center.toArray(),
   right:selected.candidates.find(c=>c.localSide!==selected.localSide).center.toArray()};
  this.scoring={...window.learningScene.scoring,version:SPECIFIED_EYE_SCORING_VERSION,
   expected_eye:PANDA_EXPECTED_EYE,side_definition:'body-labelled-iris-midplane'};
  return {eye:eyeSurface().toArray(),scoring:this.scoring,condition:{...condition,person}};
 },
 geometry(){
  const frames=[];
  for(const open of [1,.5,0]){setOpen(open);scene.updateMatrixWorld(true);frames.push(posedGeometry(vrm.scene));}
  setOpen(1);scene.updateMatrixWorld(true);
  return {person_frames:frames,chair:posedGeometry(chairRoot),eye:eyeSurface().toArray(),scoring:this.scoring};
 },
 async setup(shapes){
  const {OBJLoader}=await import('three/addons/loaders/OBJLoader.js');
  const loader=new OBJLoader();
  for(const shape of shapes){
   let group=pandaLinks.get(shape.link);
   if(!group){group=new THREE.Group();pandaLinks.set(shape.link,group);pandaRoot.add(group);}
   let object;
   const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...shape.color.slice(0,3)),roughness:.55});
   if(shape.kind==='mesh'){
    object=await loader.loadAsync('/panda-assets/'+shape.mesh);
    object.scale.set(...shape.dimensions);
    object.traverse(m=>{if(m.isMesh)m.material=material;});
   }else if(shape.kind==='box')object=new THREE.Mesh(new THREE.BoxGeometry(...shape.dimensions),material);
   else if(shape.kind==='sphere')object=new THREE.Mesh(new THREE.SphereGeometry(shape.dimensions[0],16,12),material);
   else throw Error('未登録の描画形状: '+shape.kind);
   object.position.set(...shape.position);object.quaternion.set(...shape.orientation);group.add(object);
  }
 },
 render(state,capture=true){
  this.eyeCondition=state.eye_condition||'natural';
  const open=state.eye_condition==='closed'?0:state.eye_condition==='half-open'?.5:blink(state.time+learningPhase);
  setOpen(open);scene.updateMatrixWorld(true);
  for(const link of state.links){
   const group=pandaLinks.get(link.link);if(!group)throw Error('リンク描画未登録');
   group.position.copy(physicsPoint(link.position));
   group.quaternion.copy(pandaFrame).multiply(new THREE.Quaternion(...link.orientation));
  }
  renderer.render(scene,camera);
  if(!capture)return {open};
  learningRenderer.render(scene,learningCamera);const left=learningRenderer.domElement.toDataURL('image/png').split(',')[1];
  learningRenderer.render(scene,pandaCameraRight);const right=learningRenderer.domElement.toDataURL('image/png').split(',')[1];
  return {open,left,right};
 },
 pressureSamples(samples){
  return samples.map(sample=>({...sample,open:this.eyeCondition==='closed'?0:this.eyeCondition==='half-open'?.5:blink(sample.time+learningPhase)}));
 },
 score(start,end,open){
  const hit=window.learningScene.surfaceHit(start,end);
  return hit?{...hit,...classifySpecifiedEye({hit,openness:open,annotation:anatomyAnnotation,
   eyes:this.scoringEyes,expectedEye:PANDA_EXPECTED_EYE})}:null;
 },
 scoreSegments(segments){
  const hits=[];
  for(const segment of segments){
   if(this.scored.has(segment.id))continue;
   const open=this.eyeCondition==='closed'?0:this.eyeCondition==='half-open'?.5:blink(segment.time+learningPhase);setOpen(open);scene.updateMatrixWorld(true);
   const hit=this.score(segment.start,segment.end,open);
   if(hit){this.scored.add(segment.id);hits.push({...hit,id:segment.id,time:segment.time,open});}
  }
  return hits;
 }
};
window.pandaExperiment.reset();
