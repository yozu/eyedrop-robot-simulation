import * as THREE from 'three';

const HEAD_LOCAL_SIDES=[1,-1];
const SIDE_EPSILON=1e-9;

export function irisCenter(meshes,head,surfaces=null,side=1){
 if(!HEAD_LOCAL_SIDES.includes(side))throw new Error('虹彩の頭ローカル側が不正です');
 const bounds=new THREE.Box3();
 const inverseHead=head.matrixWorld.clone().invert();
 let count=0;
 for(const mesh of meshes){
  const applicable=surfaces===null?null:surfaces.filter(surface=>surface.mesh===mesh.name);
  if(applicable?.length===0)continue;
  if(applicable?.some(surface=>surface.uv_polygon))throw new Error('UV領域を使う眼位置の計算は未対応です');
  if(mesh.isSkinnedMesh)mesh.skeleton.update();
  // 虹彩と顔が頂点バッファを共有していても、虹彩が参照する頂点だけを読む。
  const geometry=mesh.geometry,index=geometry.index;
  let indices;
  if(applicable===null){
   indices=index?new Set(index.array):Array.from({length:geometry.attributes.position.count},(_,i)=>i);
  }else{
   indices=new Set();
   const count=index?index.count:geometry.attributes.position.count;
   for(let offset=0;offset<count;offset+=3){
    let material=mesh.material;
    if(Array.isArray(material)){
     const group=geometry.groups.find(group=>offset>=group.start&&offset<group.start+group.count);
     if(!group)throw new Error('三角形の材質が定義されていません');
     material=material[group.materialIndex];
    }
    if(!applicable.some(surface=>surface.material===material.name&&(!surface.triangles||surface.triangles.includes(offset/3))))continue;
    for(let vertex=0;vertex<3;vertex++)indices.add(index?index.getX(offset+vertex):offset+vertex);
   }
  }
  for(const index of indices){
   const point=mesh.getVertexPosition(index,new THREE.Vector3());mesh.localToWorld(point);
   if(point.clone().applyMatrix4(inverseHead).x*side>0){bounds.expandByPoint(point);count++;}
  }
 }
 if(!count)throw new Error('対象眼の虹彩メッシュが見つかりません');
 return bounds.getCenter(new THREE.Vector3());
}

// 新しい教師と指定眼採点で明示的に使う。旧デフォルトと旧採点は変更しない。
export function bodyLeftIrisCenter(meshes,head,surfaces,leftUpperArm,rightUpperArm){
 if(!Array.isArray(surfaces)||surfaces.length===0)throw new Error('虹彩注釈がありません');
 if(!leftUpperArm||!rightUpperArm||![...leftUpperArm,...rightUpperArm].every(Number.isFinite))throw new Error('身体の左右参照がありません');
 const axis=leftUpperArm.clone().sub(rightUpperArm);
 if(axis.length()<=SIDE_EPSILON)throw new Error('身体の左右参照が重なっています');
 axis.normalize();
 const candidates=HEAD_LOCAL_SIDES.map(localSide=>{
  const center=irisCenter(meshes,head,surfaces,localSide);
  return {localSide,center,projection:center.dot(axis)};
 });
 if(Math.abs(candidates[0].projection-candidates[1].projection)<=SIDE_EPSILON)throw new Error('身体に対する眼の左右が曖昧です');
 const selected=candidates[0].projection>candidates[1].projection?candidates[0]:candidates[1];
 return {...selected,candidates,bodyLeftAxis:axis};
}
