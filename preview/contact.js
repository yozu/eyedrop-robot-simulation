import * as THREE from 'three';

export const CONTACT_MARGIN=.001;
const DROP_PATH_RADIUS=.001;
const EYE_TARGET_ALLOWANCE=.004;
const DESCENT={start:.08,goal:.02,step:.002};
const posedCache=new WeakMap();

export function chooseDescentHeight(blocked){
 if(blocked(DESCENT.start))throw new Error('待機位置で人物に接触します');
 let safe=DESCENT.start;
 const steps=Math.round((DESCENT.start-DESCENT.goal)/DESCENT.step);
 for(let i=1;i<=steps;i++){
  const height=Number((DESCENT.start-i*DESCENT.step).toFixed(3));
  if(blocked(height))return Math.min(DESCENT.start,safe+DESCENT.step);
  safe=height;
 }
 return DESCENT.goal;
}

function posedGeometry(mesh){
 if(mesh.isSkinnedMesh)mesh.skeleton.update();
 const signature=[mesh.geometry.uuid,mesh.geometry.attributes.position.version,mesh.geometry.index?.version,
  ...mesh.matrixWorld.elements,...(mesh.skeleton?.boneMatrices||[]),...(mesh.morphTargetInfluences||[])];
 const old=posedCache.get(mesh);
 if(old&&signature.length===old.signature.length&&signature.every((v,i)=>v===old.signature[i]))return old;
 const index=mesh.geometry.index;
 const ids=index?new Set(index.array):Array.from({length:mesh.geometry.attributes.position.count},(_,i)=>i);
 const vertices=new Map(),bounds=new THREE.Box3();
 for(const id of ids){
  const point=mesh.getVertexPosition(id,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
  vertices.set(id,point);bounds.expandByPoint(point);
 }
 const data={signature,vertices,bounds};posedCache.set(mesh,data);return data;
}

export function dropPathVolume(nozzle,eye){
 const start=new THREE.Vector3(...nozzle),target=new THREE.Vector3(...eye);
 const end=target.clone().addScaledVector(start.clone().sub(target).normalize(),EYE_TARGET_ALLOWANCE);
 return {name:'滴の経路',box:new THREE.Box3().setFromPoints([start,end]).expandByScalar(DROP_PATH_RADIUS)};
}
// ロボット部品の移動前後を包む箱と、変形後の人体三角形を照合する。
// 回転しない現行の直動機構向け。箱は保守的な近似で、反力は計算しない。
export function detectContact(person,volumes){
 person.updateWorldMatrix(true,true);
 const triangle=new THREE.Triangle();
 let contact=null;
 person.traverse(mesh=>{
  if(contact||!mesh.isMesh||!mesh.visible)return;
  const {bounds,vertices}=posedGeometry(mesh);
  const candidates=volumes.filter(v=>v.box.intersectsBox(bounds));
  if(!candidates.length)return;
  const index=mesh.geometry.index;
  const count=index?index.count:mesh.geometry.attributes.position.count;
  for(let i=0;i<count&&!contact;i+=3){
   for(let j=0;j<3;j++){
    const vertex=index?index.getX(i+j):i+j;
    [triangle.a,triangle.b,triangle.c][j].copy(vertices.get(vertex));
   }
   for(const v of candidates){
    if(v.box.intersectsTriangle(triangle)){
     contact={part:v.name,mesh:mesh.name||'人物',kind:'swept-box-triangle'};
     break;
    }
   }
  }
 });
 return contact;
}

export function contactVolumes(parts,previous=[]){
 return parts.map(({name,object},i)=>{
  object.updateWorldMatrix(true,true);
  const current=new THREE.Box3().setFromObject(object,true);
  const box=current.clone();
  if(previous[i])box.union(previous[i].current);
  box.expandByScalar(CONTACT_MARGIN);
  return {name,box,current};
 });
}
