import * as THREE from 'three';

// 採点専用。変形後の三角形と滴中心の線分が最初に交差する場所を返す。
// 滴の半径・液体の広がり・人体の反力はこの処理では計算しない。
export function firstSurfaceHit(person,start,end,visualSurfaces=[]){
 const length=start.distanceTo(end);
 if(!Number.isFinite(length))throw new Error('滴の線分が不正です');
 if(length===0)return null;
 const ray=new THREE.Ray(start.clone(),end.clone().sub(start).normalize());
 const triangle=new THREE.Triangle(),point=new THREE.Vector3(),barycentric=new THREE.Vector3();
 let nearest=null;
 person.updateWorldMatrix(true,true);
 person.traverseVisible(mesh=>{
  if(!mesh.isMesh)return;
  if(mesh.isSkinnedMesh)mesh.skeleton.update();
  const geometry=mesh.geometry,index=geometry.index,positions=geometry.attributes.position;
  const count=index?index.count:positions.count;
  const vertices=new Map(),bounds=new THREE.Box3();
  for(let i=0;i<count;i++){
   const id=index?index.getX(i):i;
   if(vertices.has(id))continue;
   const vertex=mesh.getVertexPosition(id,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
   vertices.set(id,vertex);bounds.expandByPoint(vertex);
  }
  if(!ray.intersectsBox(bounds))return;
  for(let i=0;i<count;i+=3){
   const ids=[0,1,2].map(offset=>index?index.getX(i+offset):i+offset);
   triangle.set(...ids.map(id=>vertices.get(id)));
   if(!ray.intersectTriangle(triangle.a,triangle.b,triangle.c,false,point))continue;
   const distance=start.distanceTo(point);
   if(distance>length||(nearest&&distance>=nearest.distance))continue;
   let material=mesh.material;
   if(Array.isArray(material)){
    const group=geometry.groups.find(group=>i>=group.start&&i<group.start+group.count);
    if(!group)throw new Error('三角形の材質が定義されていません');
    material=material[group.materialIndex];
   }
   // 素材ごとに確認した描画専用面だけを採点から除く。透明な髪などは残す。
   if(visualSurfaces.some(surface=>surface.mesh===mesh.name&&surface.material===material.name))continue;
   triangle.getBarycoord(point,barycentric);
   let uv=null;
   if(geometry.attributes.uv){
    const attribute=geometry.attributes.uv;
    uv=[0,1].map(axis=>ids.reduce((sum,id,j)=>sum+
     (axis===0?attribute.getX(id):attribute.getY(id))*barycentric.getComponent(j),0));
   }
   nearest={distance,point:point.toArray(),mesh:mesh.name,material:material.name,
    triangle:i/3,barycentric:barycentric.toArray(),uv};
  }
 });
 return nearest;
}
