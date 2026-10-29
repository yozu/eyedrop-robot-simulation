import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import * as THREE from './node_modules/three/build/three.module.js';
const target=process.env.PANDA_EYE_SOURCE_PREVIEW?pathToFileURL(path.resolve(process.env.PANDA_EYE_SOURCE_PREVIEW,'eye-target.js')):new URL('./eye-target.js',import.meta.url);
const source=readFileSync(target,'utf8')
 .replace("'three'",JSON.stringify(new URL('./node_modules/three/build/three.module.js',import.meta.url).href));
const {irisCenter}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const geometry=new THREE.BufferGeometry();
geometry.setAttribute('position',new THREE.Float32BufferAttribute([1,0,0, 1,2,0, 1,0,2, 100,100,100],3));
geometry.setIndex([0,1,2]);
const mesh=new THREE.Mesh(geometry);
const head=new THREE.Group();head.updateMatrixWorld(true);mesh.updateMatrixWorld(true);
assert.deepEqual(irisCenter([mesh],head).toArray(),[1,1,1]);
mesh.position.set(0,3,0);mesh.updateMatrixWorld(true);
assert.deepEqual(irisCenter([mesh],head).toArray(),[1,4,1]);
assert.throws(()=>irisCenter([],head),/虹彩/);
console.log('虹彩の未参照頂点を除外・座標変換・欠損時停止: 合格');
