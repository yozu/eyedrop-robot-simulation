import * as THREE from 'three';

const TARGET_TOLERANCE_M=.001;
const IMAGE_SIDE=256;
const LINE_PIXEL_RADIUS=1;

// 診断用。人物の教師点への視線が、描画される物体で遮られるか調べる。
export function cameraVisibility(camera,point,groups){
 if(!point?.isVector3||![point.x,point.y,point.z].every(Number.isFinite))throw new Error('視認点が不正です');
 camera.updateMatrixWorld(true);
 const origin=camera.getWorldPosition(new THREE.Vector3()),delta=point.clone().sub(origin),distance=delta.length();
 const projected=point.clone().project(camera);
 const inFrame=distance>0&&Math.abs(projected.x)<=1&&Math.abs(projected.y)<=1&&projected.z>=-1&&projected.z<=1;
 if(!inFrame)return {inFrame:false,visible:false,firstGroup:null,targetDistance:distance};
 const ray=new THREE.Raycaster(origin,delta.normalize(),camera.near,distance+TARGET_TOLERANCE_M);
 if(!camera.isPerspectiveCamera)throw new Error('透視カメラ以外の視認性は未対応です');
 // CAD輪郭線の既定1m判定を使わず、対象までの最大深度で1画素の半径へ換算する。
 ray.params.Line.threshold=2*distance*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*LINE_PIXEL_RADIUS/IMAGE_SIDE;
 ray.layers.mask=camera.layers.mask;
 const hits=[];
 for(const {root,label} of groups){
  root.updateMatrixWorld(true);
  for(const hit of ray.intersectObject(root,true)){
   let shown=true;for(let node=hit.object;node;node=node.parent)shown&&=node.visible;
   if(shown)hits.push({...hit,label});
  }
 }
 hits.sort((a,b)=>a.distance-b.distance);const first=hits[0];
 return {inFrame:true,visible:!!first&&first.label==='person'&&Math.abs(first.distance-distance)<=TARGET_TOLERANCE_M,
  firstGroup:first?.label??null,firstMesh:first?.object.name??null,firstDistance:first?.distance??null,targetDistance:distance,
  firstType:first?.object.type??null,firstDistanceToRay:first?.distanceToRay??null,
  lineThresholdM:ray.params.Line.threshold,
  firstProjected:first?first.point.clone().project(camera).toArray():null,
  pixelNormalized:[(projected.x+1)/2,(1-projected.y)/2]};
}
