import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as THREE from 'three';
import {pandaCondition} from './panda-condition.mjs';
import {firstSurfaceHit} from './anatomy-contact.mjs';
import {classifySpecifiedEye,SPECIFIED_EYE_SCORING_VERSION} from './anatomy-score.mjs';

// ブラウザ用.jsはパッケージ設定を変更せず、同じESMソースで試験する。
const eyeSource=readFileSync(new URL('./eye-target.js',import.meta.url),'utf8')
 .replace("from 'three'",`from '${import.meta.resolve('three')}'`);
const {bodyLeftIrisCenter}=await import('data:text/javascript;base64,'+Buffer.from(eyeSource).toString('base64'));

function runtime(){
 const scene=new THREE.Scene(),root=new THREE.Group(),head=new THREE.Group();
 scene.add(root);root.add(head);
 const positions=[];
 for(const x of [.03,-.03])positions.push(x-.005,-.005,.1,x+.005,-.005,.1,x,.005,.1);
 const geometry=new THREE.BufferGeometry();
 geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
 const material=new THREE.MeshBasicMaterial();material.name='iris';
 const mesh=new THREE.Mesh(geometry,material);mesh.name='eyes';head.add(mesh);
 const left=new THREE.Object3D(),right=new THREE.Object3D();
 left.position.x=.2;right.position.x=-.2;root.add(left,right);
 const bones={head,leftUpperArm:left,rightUpperArm:right};
 const vrm={scene:root,humanoid:{getRawBoneNode:name=>bones[name]},update:()=>{}};
 const context={THREE,pandaCondition,bodyLeftIrisCenter,classifySpecifiedEye,SPECIFIED_EYE_SCORING_VERSION,
  renderer:{setAnimationLoop:()=>{}},learningRenderer:{setSize:()=>{}},learningCamera:new THREE.PerspectiveCamera(),
  paused:false,manual:false,scene,vrm,head,basePersonPosition:new THREE.Vector3(),HEAD_POSE:{head:0},poseSign:1,
  bone:name=>bones[name],learningPhase:0,setOpen:()=>{},person:'fixture',blink:()=>1,
  anatomyAnnotation:{eye_surfaces:[{mesh:'eyes',material:'iris'}]},
  // 教師位置を意図的に反対眼へ置いても、採点は実際の表面と身体左右から決まる。
  eyeSurface:()=>new THREE.Vector3(-.03,0,.1),window:{learningScene:{scoring:{version:'first-surface-open-eye-v2'},
   surfaceHit:(start,end)=>firstSurfaceHit(root,new THREE.Vector3(...start),new THREE.Vector3(...end)),
   classifyHit:()=>{throw Error('旧採点へ接続されました');}}}};
 const source=readFileSync(new URL('./panda-runtime.js',import.meta.url),'utf8').replace(/^import .*;\r?$/gm,'');
 vm.runInNewContext(source,context);
 return context.window.pandaExperiment;
}

test('Panda本体は実三角形の反対眼を拒否し、誤った教師位置を採点に使わない',()=>{
 const api=runtime();
 assert.equal(api.score([.03,0,.2],[.03,0,0],1).result,'target');
 assert.equal(api.score([-.03,0,.2],[-.03,0,0],1).result,'wrong-eye');
 assert.equal(api.scoring.version,SPECIFIED_EYE_SCORING_VERSION);
 assert.equal(api.scoring.expected_eye,'left');
});

test('Panda resetの人物移動後も採点参照を更新し、滴の重複採点をしない',()=>{
 const api=runtime();api.reset(.2,0,.4,-.5);
 const segments=[{id:1,time:0,start:[-.47,.4,.4],end:[-.47,.4,.2]},
  {id:2,time:0,start:[-.53,.4,.4],end:[-.53,.4,.2]}];
 const hits=api.scoreSegments(segments);
 assert.equal(hits.length,2);assert.equal(hits[0].result,'target');assert.equal(hits[1].result,'wrong-eye');
 assert.equal(api.scoreSegments(segments).length,0);
 api.eyeCondition='half-open';
 assert.equal(api.scoreSegments([{...segments[0],id:3}])[0].result,'not-open');
});
