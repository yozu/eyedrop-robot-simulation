import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {readFile} from 'node:fs/promises';
const threeUrl=new URL('./node_modules/three/build/three.module.js',import.meta.url).href;
const asModule=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const contactSource=(await readFile(new URL('./contact.js',import.meta.url),'utf8'))
 .replace("'three'",JSON.stringify(threeUrl));
const diagnosticSource=(await readFile(new URL('./diagnostic.js',import.meta.url),'utf8'))
 .replace("'three'",JSON.stringify(threeUrl))
 .replace("'./contact.js'",JSON.stringify(asModule(contactSource)));
const {createDiagnostic}=await import(asModule(diagnosticSource));

test('学習描画は端点間の横断接触を保持し、次の試行でリセットする',()=>{
 const previousDocument=globalThis.document;
 globalThis.document={getElementById:()=>({})};
 try{
  const scene=new THREE.Scene(),person=new THREE.Group();
  const face=new THREE.Mesh(new THREE.BoxGeometry(.001,.02,.02),new THREE.MeshBasicMaterial());
  face.position.set(0,1.03,0);person.add(face);scene.add(person);
  const diagnostic=createDiagnostic({scene,person,eye:()=>new THREE.Vector3(),apply:()=>{}});
  const state=(frame,x)=>({frame,time:frame/20,nozzle:[x,1,0],
   observation:{state:[x,.1,.1,0]},drops:[]});
  assert.equal(diagnostic.renderLearning(state(0,-.1),1),null);
  const contact=diagnostic.renderLearning(state(1,.1),1);
  assert.ok(contact,'両端が非接触でも途中の人物表面を検出する');
  assert.equal(diagnostic.renderLearning(state(1,.1),1),contact,
   '採点と次の推論前の重複描画で接触を消さない');
  assert.equal(diagnostic.renderLearning(state(0,.1),1),null,
   '前試行からの移動経路を判定しない');
 }finally{globalThis.document=previousDocument;}
});
