import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from './node_modules/three/build/three.module.js';
import {firstSurfaceHit} from './anatomy-contact.mjs';

function surface(z,name){
 const mesh=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial());
 mesh.position.z=z;mesh.material.name=name;return mesh;
}
test('閉じたまぶたが眼より手前なら最初の接触はまぶた',()=>{
 const person=new THREE.Group(),eye=surface(0,'eye'),lid=surface(.1,'lid');
 person.add(eye,lid);
 const a=new THREE.Vector3(0,0,1),b=new THREE.Vector3(0,0,-1);
 assert.equal(firstSurfaceHit(person,a,b).material,'lid');
 lid.visible=false;
 assert.equal(firstSurfaceHit(person,a,b).material,'eye');
});
test('眼から外れた滴は顔に当たり、線分の先には当たらない',()=>{
 const person=new THREE.Group(),eye=surface(.1,'eye'),face=surface(0,'face');
 eye.scale.set(.1,.1,1);person.add(face,eye);
 const a=new THREE.Vector3(.3,0,1);
 assert.equal(firstSurfaceHit(person,a,new THREE.Vector3(.3,0,-1)).material,'face');
 assert.equal(firstSurfaceHit(person,a,new THREE.Vector3(.3,0,.2)),null);
});
test('形状変化を反映し、単なる水平目標面の通過では成功しない',()=>{
 const person=new THREE.Group(),eye=surface(0,'eye');
 person.add(eye);
 const a=new THREE.Vector3(0,0,1),b=new THREE.Vector3(0,0,-1);
 assert.ok(firstSurfaceHit(person,a,b));
 eye.position.x=2;
 assert.equal(firstSurfaceHit(person,a,b),null);
});
test('まばたき用の頂点変形と接触地点のUVを反映する',()=>{
 const person=new THREE.Group(),eye=surface(0,'eye');
 const count=eye.geometry.attributes.position.count;
 const deltas=new Float32Array(count*3);
 for(let i=0;i<count;i++)deltas[i*3]=2;
 eye.geometry.morphAttributes.position=[new THREE.Float32BufferAttribute(deltas,3)];
 eye.geometry.morphTargetsRelative=true;eye.updateMorphTargets();person.add(eye);
 const a=new THREE.Vector3(0,0,1),b=new THREE.Vector3(0,0,-1);
 assert.deepEqual(firstSurfaceHit(person,a,b).uv,[.5,.5]);
 eye.morphTargetInfluences[0]=1;
 assert.equal(firstSurfaceHit(person,a,b),null);
});
