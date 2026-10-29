import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const C={skin:0xc99172,lip:0x975950,hair:0x302d28,shirt:0x547c78,pants:0x344553,
 chair:0x283e42,metal:0x869895,sole:0x243232,floor:0xe8eee7};
const N={segments:48,rings:32,eyeRadius:.018,lidRadius:.0192,close:.095,hold:.04,open:.17,
 blinkStarts:[1.1,3.65,4.12,7.3],cycle:9,torsoTilt:-.35,headTilt:-.48};
const scene=new THREE.Scene();scene.background=new THREE.Color(C.floor);
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
document.body.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(34,innerWidth/innerHeight,.005,30);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const mat=(color,roughness=.65)=>new THREE.MeshStandardMaterial({color,roughness});
const skin=mat(C.skin),shirt=mat(C.shirt),pants=mat(C.pants),hair=mat(C.hair),chair=mat(C.chair);
function mesh(geo,material,parent=scene){const m=new THREE.Mesh(geo,material);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function ellipsoid(pos,scale,material,parent=scene){
 const m=mesh(new THREE.SphereGeometry(1,N.segments,N.rings),material,parent);m.position.set(...pos);m.scale.set(...scale);return m;
}
function box(pos,size,material,parent=scene){
 const m=mesh(new THREE.BoxGeometry(...size),material,parent);m.position.set(...pos);
 return m;
}
function bar(a,b,r,material,parent=scene){
 const va=new THREE.Vector3(...a),vb=new THREE.Vector3(...b),d=vb.clone().sub(va);
 const m=mesh(new THREE.CylinderGeometry(r,r,d.length(),24),material,parent);
 m.position.copy(va.add(vb).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());return m;
}
function line(points,r,material,parent){
 const curve=new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p)));
 return mesh(new THREE.TubeGeometry(curve,40,r,8,false),material,parent);
}
scene.add(new THREE.HemisphereLight(0xffffff,0x7e9890,2.1));
function light(pos,power,size){const l=new THREE.DirectionalLight(0xfff4e6,power);l.position.set(...pos);l.castShadow=true;l.shadow.mapSize.set(2048,2048);l.shadow.camera.left=-2;l.shadow.camera.right=2;l.shadow.camera.top=2;l.shadow.camera.bottom=-2;l.shadow.bias=-.0003;scene.add(l);}
light([2,4,3],3);scene.add(new THREE.AmbientLight(0xffffff,.25));
const floor=mesh(new THREE.PlaneGeometry(20,20),mat(C.floor));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;

// 五本脚・キャスター・昇降支柱・背もたれ・ヘッドレスト。
const metal=mat(C.metal,.35);
bar([0,.09,0],[0,.49,0],.035,metal);
for(let i=0;i<5;i++){const a=i*Math.PI*2/5;const end=[Math.sin(a)*.34,.07,Math.cos(a)*.34];
 bar([0,.13,0],end,.022,metal);ellipsoid([end[0],.04,end[2]],[.033,.04,.042],chair);}
ellipsoid([0,.51,.02],[.285,.065,.26],chair);
const back=new THREE.Group();scene.add(back);back.position.set(0,.54,-.15);back.rotation.x=N.torsoTilt;
ellipsoid([0,.29,-.05],[.255,.32,.055],chair,back);
ellipsoid([0,.69,-.055],[.15,.10,.055],chair,back);
for(let i=-1;i<=1;i+=2){
 bar([i*.28,.49,0],[i*.28,.76,0],.018,metal);
 ellipsoid([i*.28,.77,.08],[.055,.028,.18],chair);
}

// 座位の体幹・脚・靴。左右は同じ関節位置から組み立てる。
const torso=new THREE.Group();scene.add(torso);torso.position.set(0,.59,0);torso.rotation.x=N.torsoTilt;
ellipsoid([0,.05,0],[.19,.11,.13],pants,torso);
ellipsoid([0,.27,0],[.205,.27,.115],shirt,torso);
ellipsoid([0,.46,0],[.215,.08,.1],shirt,torso);
bar([0,.48,0],[0,.62,0],.045,skin,torso);
for(const side of [-1,1]){
 ellipsoid([side*.105,.60,.24],[.105,.085,.26],pants);
 bar([side*.105,.57,.44],[side*.105,.14,.5],.073,pants);
 ellipsoid([side*.105,.075,.58],[.085,.06,.16],mat(C.sole));
 const shoulder=[side*.20,.43,0],elbow=[side*.275,.20,.13],wrist=[side*.275,.20,.34];
 bar(shoulder,elbow,.063,shirt,torso);ellipsoid(elbow,[.062,.065,.062],shirt,torso);
 bar(elbow,wrist,.043,skin,torso);ellipsoid(wrist,[.046,.022,.075],skin,torso);
 for(let f=0;f<4;f++)bar([side*.275+(f-1.5)*.014,.2,.37],[side*.275+(f-1.5)*.014,.194,.425],.008,skin,torso);
}

const head=new THREE.Group();torso.add(head);head.position.set(0,.70,.005);head.rotation.x=N.headTilt;
// 顎・頬・こめかみ・頭頂を別の断面として持つ頭部メッシュ。
const controlProfile=[[-.125,.025,.032],[-.112,.047,.055],[-.09,.060,.065],[-.065,.072,.071],
 [-.035,.080,.074],[0,.082,.075],[.035,.079,.073],[.07,.077,.070],[.105,.064,.058],[.125,.035,.03],[.13,.002,.002]];
const profileCurve=new THREE.CatmullRomCurve3(controlProfile.map(p=>new THREE.Vector3(...p)));
const profile=profileCurve.getPoints(60).map(p=>p.toArray());
const verts=[],indices=[];
for(let j=0;j<profile.length;j++){const [y,rx,rz]=profile[j];for(let i=0;i<=N.segments;i++){
 const a=i/N.segments*Math.PI*2;verts.push(rx*Math.sin(a),y,rz*Math.cos(a));
 if(j<profile.length-1&&i<N.segments){const k=j*(N.segments+1)+i;indices.push(k,k+1,k+N.segments+1,k+1,k+N.segments+2,k+N.segments+1);}
}}
const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));geom.setIndex(indices);geom.computeVertexNormals();
mesh(geom,skin,head);
for(const s of [-1,1]){
 ellipsoid([s*.082,0,-.005],[.015,.034,.018],skin,head);
 ellipsoid([s*.088,.002,.007],[.007,.019,.006],mat(0xad735d),head);
 
}
// 鼻梁・鼻翼・唇・眉。眼球を貼るだけの顔にしない。
ellipsoid([0,.004,.077],[.010,.033,.014],skin,head);
ellipsoid([0,-.018,.091],[.015,.011,.016],skin,head);
for(const s of [-1,1]){ellipsoid([s*.011,-.022,.087],[.008,.008,.010],skin,head);ellipsoid([s*.010,-.027,.095],[.004,.002,.003],mat(0x724c42),head);}
line([[-.026,-.061,.068],[-.012,-.058,.076],[0,-.060,.078],[.012,-.058,.076],[.026,-.061,.068]],.003,mat(C.lip),head);
line([[-.024,-.064,.068],[0,-.067,.077],[.024,-.064,.068]],.0035,mat(0xb77567),head);
line([[-.022,-.062,.072],[0,-.063,.079],[.022,-.062,.072]],.0008,mat(0x69423a),head);
for(const s of [-1,1])line([[s*.014,.054,.074],[s*.032,.060,.073],[s*.052,.055,.064]],.003, hair,head);
const haircap=mesh(new THREE.SphereGeometry(1,48,24,0,Math.PI*2,0,Math.PI*.335),hair,head);
haircap.scale.set(.083,.125,.080);haircap.position.set(0,.014,-.008);
for(const s of [-1,1])ellipsoid([s*.074,.055,-.024],[.012,.055,.055],hair,head);

const lids=[],rims=[];
function lidGeometry(top,amount){
 const v=[],ix=[],lonCount=40,latCount=14;
 for(let j=0;j<=latCount;j++)for(let i=0;i<=lonCount;i++){
  const lon=-Math.PI/2+i/lonCount*Math.PI;
  const edge=Math.asin(amount*(top?.61:-.40)*Math.cos(lon));
  const lat=edge+( (top?Math.PI/2:-Math.PI/2)-edge)*j/latCount;
  v.push(N.lidRadius*Math.cos(lat)*Math.sin(lon),N.lidRadius*Math.sin(lat),N.lidRadius*Math.cos(lat)*Math.cos(lon));
  if(i<lonCount&&j<latCount){const k=j*(lonCount+1)+i;ix.push(k,k+1,k+lonCount+1,k+1,k+lonCount+2,k+lonCount+1);}
 }
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(v,3));g.setIndex(ix);g.computeVertexNormals();return g;
}
for(const s of [-1,1]){
 const eye=new THREE.Group();head.add(eye);eye.position.set(s*.032,.025,.063);eye.scale.setScalar(.80);
 ellipsoid([0,0,0],[N.eyeRadius,N.eyeRadius,N.eyeRadius],mat(0xf4eee5,.3),eye);
 const iris=mesh(new THREE.SphereGeometry(.01815,32,16,0,Math.PI*2,0,Math.asin(.0075/.01815)),mat(0x587a70,.3),eye);iris.rotation.x=Math.PI/2;
 const pupil=mesh(new THREE.SphereGeometry(.0183,32,12,0,Math.PI*2,0,Math.asin(.0031/.0183)),mat(0x122629,.18),eye);pupil.rotation.x=Math.PI/2;
 ellipsoid([-.002,.002,.0182],[.0011,.0011,.00015],mat(0xffffff,.05),eye);
 const rimMaterial=mat(0x98634f);
 const rim=line([[-.018,0,0],[0,.006,.018],[.018,0,0]],.00045,rimMaterial,eye);rims.push(rim);
 for(const top of [true,false]){
  const material=skin.clone();material.side=THREE.DoubleSide;
  const lid=mesh(lidGeometry(top,1),material,eye);lids.push({lid,top});
 }
}
let openness=1,last=-1,paused=false,manual=false,clock=0,view='wide';
function setOpen(value){
 openness=value;
 if(Math.abs(value-last)<.0001)return;
 for(const {lid,top} of lids){lid.geometry.dispose();lid.geometry=lidGeometry(top,value);}
 for(const rim of rims){
 const points=[];for(let i=0;i<=40;i++){const lon=-Math.PI/2+i/40*Math.PI;const lat=Math.asin(value*.61*Math.cos(lon));points.push(new THREE.Vector3(N.lidRadius*Math.cos(lat)*Math.sin(lon),N.lidRadius*Math.sin(lat),N.lidRadius*Math.cos(lat)*Math.cos(lon)));}
 rim.geometry.dispose();rim.geometry=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),40,.00035,6,false);
 }
 last=value;
}
export function blink(t){
 const phase=t%N.cycle;
 for(const start of N.blinkStarts){let q=phase-start;
  if(q<0)continue;if(q<N.close)return (1+Math.cos(Math.PI*q/N.close))/2;
  q-=N.close;if(q<N.hold)return 0;q-=N.hold;
  if(q<N.open)return (1-Math.cos(Math.PI*q/N.open))/2;
 }
 return 1;
}
function setView(name){
 view=name;scene.updateMatrixWorld(true);
 if(name==='face'){
  const p=head.getWorldPosition(new THREE.Vector3()),q=head.getWorldQuaternion(new THREE.Quaternion());
  controls.target.copy(p);
  camera.position.copy(new THREE.Vector3(.10,.035,.58).applyQuaternion(q).add(p));
 }else{controls.target.set(0,.79,.02);camera.position.set(2.15,1.75,2.65);}
 controls.update();
}
document.getElementById('wide').onclick=()=>setView('wide');
document.getElementById('face').onclick=()=>setView('face');
document.getElementById('pause').onclick=()=>{paused=!paused;document.getElementById('pause').textContent=paused?'再生':'一時停止';};
document.getElementById('open').oninput=e=>{manual=true;setOpen(Number(e.target.value));};
document.getElementById('auto').onclick=()=>{manual=false;paused=false;};
function renderAt(t,value=null){
 clock=t;if(value!==null){manual=true;setOpen(value);}else if(!manual)setOpen(blink(t));
 document.getElementById('status').textContent=`${t.toFixed(2)}秒 ｜ 開眼量 ${(openness*100).toFixed(0)}%`;
 document.getElementById('open').value=openness;renderer.render(scene,camera);
}
window.preview={setView,renderAt,seek:(t)=>{manual=false;paused=true;renderAt(t);},pause:()=>{paused=true;},info:()=>({openness,view,lids:lids.length,meshCount:scene.children.length})};
setView('wide');let prev=performance.now();
renderer.setAnimationLoop(now=>{const dt=Math.min((now-prev)/1000,.1);prev=now;if(!paused)clock+=dt;controls.update();renderAt(clock);});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
window.previewReady=true;
