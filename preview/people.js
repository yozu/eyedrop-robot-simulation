import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from './node_modules/@pixiv/three-vrm/lib/three-vrm.module.js';

const VIEW={width:900,height:570,headOffset:.08,distance:.65,fov:32};
const scene=new THREE.Scene();
scene.background=new THREE.Color(0xedf1ec);
scene.add(new THREE.HemisphereLight(0xffffff,0x768779,2));
const light=new THREE.DirectionalLight(0xffffff,2);
light.position.set(1,2,3);scene.add(light);
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setSize(VIEW.width,VIEW.height);
document.querySelector('#view').append(renderer.domElement);
const camera=new THREE.PerspectiveCamera(VIEW.fov,VIEW.width/VIEW.height,.01,10);
const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));
const manifest=await fetch('./assets/people/manifest.json').then(r=>r.json());
const models=[
 {id:'original',credit:'Model © 2022 pixiv Inc. — VRM Public License 1.0'},
 ...manifest.models.filter(m=>m.status==='downloaded_structurally_checked')
];
const select=document.querySelector('#model');
for(const m of models)select.add(new Option(m.id,m.id));
let vrm=null;
function blink(value){
 if(!vrm)return;
 vrm.expressionManager.setValue('blink',value);
 vrm.update(0);renderer.render(scene,camera);
 document.querySelector('#blink').value=value;
}
async function load(id){
 window.people.ready=false;select.disabled=true;
 document.querySelector('#status').textContent='読込中';
 try{
  const m=models.find(item=>item.id===id);
  if(!m)throw new Error('人物候補がありません');
  const gltf=await loader.loadAsync(id==='original'?'./assets/avatar.vrm':`./assets/people/${id}.vrm`);
  if(vrm){scene.remove(vrm.scene);VRMUtils.deepDispose(vrm.scene);}
  vrm=gltf.userData.vrm;
  VRMUtils.rotateVRM0(vrm);scene.add(vrm.scene);vrm.update(0);
  const head=vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(new THREE.Vector3());
  head.y+=VIEW.headOffset;
  camera.position.copy(head).add(new THREE.Vector3(0,0,VIEW.distance));camera.lookAt(head);
  select.value=id;blink(0);
  document.querySelector('#credit').textContent=m.credit||`${m.author} — ${m.license}`;
  document.querySelector('#status').textContent=`${id}：顔と閉眼の素材確認。着座は点眼診断画面の人物選択から確認できます。`;
  window.people.ready=true;
 }catch(error){
  document.querySelector('#status').textContent=`読込失敗：${error.message}`;throw error;
 }finally{select.disabled=false;}
}
window.people={load,blink,ready:false,ids:models.map(m=>m.id)};
select.addEventListener('change',()=>load(select.value));
document.querySelector('#blink').addEventListener('input',e=>blink(Number(e.target.value)));
await load(models[0].id);
