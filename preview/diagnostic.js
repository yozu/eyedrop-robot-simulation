import * as THREE from 'three';
import {detectContact,contactVolumes,dropPathVolume,chooseDescentHeight} from './contact.js';

const DISPLAY_HZ=30,DROP_DISPLAY_RADIUS=.002;
const TARGET_RADIUS=.006,POST_OFFSET=.32,TOP_CLEARANCE=.22;
const COLORS={frame:0x718680,head:0xd7e0da,nozzle:0x263d39,drop:0x168bc2,target:0xdb983d};
const LABELS={moving:'目の上へ移動中',waiting:'開眼の継続を待っています',squeezing:'ボトルの側面を押しています',falling:'1滴を放出',finished:'着滴判定が完了'};
// ND15本体の公表外形寸法を基準にする。口部の細部と押圧量は描画用の近似。
const BOTTLE={width:.0393,depth:.0159,bodyHeight:.0497,tipHeight:.008,
 neckRadius:.0045,tipRadius:.0027,padY:.034,padHalf:.003,
 compression:.12,holderDepth:.025,segments:48,bevel:.002,labelOffset:.0002};
const RESULTS={target:'目標領域',lid:'閉じたまぶた',face:'目標の外'};
export function createDiagnostic({scene,eye,apply,person}){
 const root=new THREE.Group();root.visible=false;scene.add(root);
 const material=color=>new THREE.MeshStandardMaterial({color,roughness:.45});
 const cube=(size,color)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(...size),material(color));root.add(m);m.castShadow=true;return m;};
 const post=cube([.025,1,.025],COLORS.frame),rail=cube([.42,.018,.018],COLORS.frame);
 const slide=cube([.035,.02,.07],COLORS.head),stem=cube([.01,.1,.01],COLORS.frame);
 // 開栓済みの柔らかい目薬容器を逆さに保持する。寸法は模式モデル用。
 const bottle=new THREE.Group();root.add(bottle);
 const halfWidth=BOTTLE.width/2,halfDepth=BOTTLE.depth/2;
 const shoulder=BOTTLE.tipHeight+.012,base=BOTTLE.tipHeight+BOTTLE.bodyHeight;
 const outline=new THREE.Shape();
 outline.moveTo(-BOTTLE.neckRadius,BOTTLE.tipHeight+.007);
 outline.quadraticCurveTo(-BOTTLE.neckRadius,shoulder,-halfWidth+BOTTLE.bevel,shoulder+.004);
 outline.lineTo(-halfWidth+BOTTLE.bevel,base-.008);
 outline.quadraticCurveTo(-halfWidth+BOTTLE.bevel,base-BOTTLE.bevel,-halfWidth+.009,base-BOTTLE.bevel);
 outline.lineTo(halfWidth-.009,base-BOTTLE.bevel);
 outline.quadraticCurveTo(halfWidth-BOTTLE.bevel,base-BOTTLE.bevel,halfWidth-BOTTLE.bevel,base-.008);
 outline.lineTo(halfWidth-BOTTLE.bevel,shoulder+.004);
 outline.quadraticCurveTo(BOTTLE.neckRadius,shoulder,BOTTLE.neckRadius,BOTTLE.tipHeight+.007);
 outline.closePath();
 const geo=new THREE.ExtrudeGeometry(outline,{depth:BOTTLE.depth-2*BOTTLE.bevel,bevelEnabled:true,
  bevelThickness:BOTTLE.bevel,bevelSize:BOTTLE.bevel,bevelSegments:5,steps:1,curveSegments:16});
 geo.translate(0,0,-halfDepth+BOTTLE.bevel);
 const shell=new THREE.Mesh(geo,new THREE.MeshPhysicalMaterial({color:0xe4f1eb,roughness:.22,transparent:true,opacity:.75,depthWrite:false}));
 bottle.add(shell);shell.castShadow=true;
 const label=new THREE.Mesh(new THREE.PlaneGeometry(BOTTLE.width*.65,.021),material(0x488e97));
 label.position.set(0,BOTTLE.padY+.006,halfDepth+BOTTLE.labelOffset);bottle.add(label);
 // 首は円形を保ち、短く丸い中栓と小さな吐出口を描く。
 const neck=new THREE.Mesh(new THREE.CylinderGeometry(BOTTLE.neckRadius,BOTTLE.neckRadius,.01,BOTTLE.segments),material(0xe6eee7));
 neck.position.y=BOTTLE.tipHeight+.003;bottle.add(neck);
 const tipProfile=[[BOTTLE.tipRadius,0],[BOTTLE.tipRadius+.0007,.001],
  [BOTTLE.tipRadius,.002],[.002,.005],[BOTTLE.neckRadius,BOTTLE.tipHeight]];
 const tip=new THREE.Mesh(new THREE.LatheGeometry(tipProfile.map(([r,h])=>new THREE.Vector2(r,h)),BOTTLE.segments),material(0x75b4c4));bottle.add(tip);
 const tipHole=new THREE.Mesh(new THREE.CircleGeometry(.0003,20),material(COLORS.nozzle));
 tipHole.rotation.x=Math.PI/2;tipHole.position.y=-.0001;bottle.add(tipHole);
 for(const height of [.009,.012]){const thread=new THREE.Mesh(new THREE.TorusGeometry(BOTTLE.neckRadius,.0005,8,32),material(0xd1e0da));thread.rotation.x=Math.PI/2;thread.position.y=height;bottle.add(thread);}
 // 幅方向の縁ではなく、薄い胴の前後の広い面を挟む。
 const pads=[-1,1].map(sign=>{const pad=cube([.017,.016,BOTTLE.padHalf*2],COLORS.nozzle);bottle.add(pad);pad.userData.sign=sign;return pad;});
 const holder=cube([.008,.012,.05],COLORS.head);bottle.add(holder);holder.position.set(halfWidth+.008,BOTTLE.padY,0);
 const fingers=pads.map(pad=>{const finger=cube([halfWidth+.008,.007,.006],COLORS.head);bottle.add(finger);finger.userData.sign=pad.userData.sign;return finger;});
 const neckClip=new THREE.Mesh(new THREE.TorusGeometry(BOTTLE.neckRadius+.001,.001,12,32),material(COLORS.head));
 neckClip.rotation.x=Math.PI/2;neckClip.position.y=BOTTLE.tipHeight+.005;bottle.add(neckClip);
 const support=cube([.006,.024,.006],COLORS.head);bottle.add(support);support.position.set(halfWidth+.008,.023,0);
 const clipArm=cube([halfWidth+.008,.004,.006],COLORS.head);bottle.add(clipArm);clipArm.position.set((halfWidth+.008)/2,neckClip.position.y,0);
 const drop=new THREE.Mesh(new THREE.SphereGeometry(DROP_DISPLAY_RADIUS,16,12),material(COLORS.drop));root.add(drop);drop.visible=false;
 const target=new THREE.Mesh(new THREE.RingGeometry(TARGET_RADIUS*.9,TARGET_RADIUS,40),new THREE.MeshBasicMaterial({color:COLORS.target,side:THREE.DoubleSide}));target.rotation.x=-Math.PI/2;root.add(target);
 const status=document.getElementById('robot-status'),button=document.getElementById('diagnostic');
 const controls=['pause','open','auto'];
 const parts=[{name:'目薬ボトル・保持具',object:bottle},{name:'支柱',object:post},
  {name:'レール',object:rail},{name:'スライダー',object:slide},{name:'アーム',object:stem}];
 let running=false,last=null,run=null,previousVolumes=[],contact=null;
 const learningDrops=new Map();
 async function request(path,body){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!response.ok)throw new Error(`接続エラー (${response.status})`);return response.json();}
 function place(state){const [x,y,z]=state.nozzle;const [ex,ey,ez]=state.eye;
  bottle.position.set(x,y,z);
  shell.scale.z=1-BOTTLE.compression*state.squeeze;
  label.position.z=halfDepth*shell.scale.z+BOTTLE.labelOffset;
  for(const pad of pads)pad.position.set(0,BOTTLE.padY,pad.userData.sign*(halfDepth*shell.scale.z+BOTTLE.padHalf));
  for(const finger of fingers)finger.position.set((halfWidth+.008)/2,BOTTLE.padY,finger.userData.sign*(halfDepth*shell.scale.z+BOTTLE.padHalf));
  const top=ey+TOP_CLEARANCE;post.scale.y=top;post.position.set(ex+POST_OFFSET,top/2,ez);
  rail.position.set(ex+POST_OFFSET/2,top,ez);slide.position.set(x,top,z);
  const mount=y+BOTTLE.padY;
  stem.scale.y=(top-mount)/.1;stem.position.set(x+halfWidth+.008,(top+mount)/2,z);
  target.position.set(ex,ey,ez);drop.visible=state.drop!==null;
  if(state.drop)drop.position.set(...state.drop);
 }
 function show(state){
  place(state);
  apply(state);
  const volumes=contactVolumes(parts,previousVolumes);
  contact=detectContact(person,volumes);
  if(!contact&&state.status==='squeezing'){
   contact=detectContact(person,[dropPathVolume(state.nozzle,state.eye)]);
  }
  if(contact){
   root.visible=false;
   status.textContent=`接触検知で停止：${contact.part} が人物の表面に接触します。重なる配置を表示せず、次の更新を停止しました。`;
   return false;
  }
  previousVolumes=volumes;last=state;
  status.textContent=`診断：${LABELS[state.status]}${state.result?' ／ '+RESULTS[state.result]:''} ／ 先端と眼 ${(state.dispense_height*100).toFixed(1)}cm（AI未接続）`;
  return true;
 }
 function stop(){running=false;button.disabled=false;for(const id of controls)document.getElementById(id).disabled=false;}
 async function tick(){if(!running)return;
  try{const state=await request('/api/step',{run,force_closed:document.getElementById('hold-closed').checked});if(!show(state)){stop();return;}
   if(state.result){stop();return;}setTimeout(tick,1000/DISPLAY_HZ);
  }catch(error){stop();status.textContent=`停止：${error.message}。診断サーバーで開いてください。`;}
 }
 async function start(){if(running)return;running=true;button.disabled=true;previousVolumes=[];contact=null;last=null;
  for(const id of controls)document.getElementById(id).disabled=true;
  try{
   root.visible=false;
   // 経路は開眼時の形状で準備し、実行時は従来どおり開眼継続を待つ。
   apply({time:0,openness:1});
   const targetEye=eye().toArray();
   const height=chooseDescentHeight(h=>{
    place({eye:targetEye,nozzle:[targetEye[0],targetEye[1]+h,targetEye[2]],squeeze:0,drop:null});
    return detectContact(person,contactVolumes(parts));
   });
   const obstruction=detectContact(person,[dropPathVolume([targetEye[0],targetEye[1]+height,targetEye[2]],targetEye)]);
   if(obstruction)throw new Error('滴の経路が人物の表面で遮られています');
   const state=await request('/api/start',{eye:targetEye,dispense_height:height});run=state.run;
   root.visible=true;if(!show(state)){stop();return;}await tick();
  }
  catch(error){stop();status.textContent=`停止：${error.message}`;}
 }
 button.onclick=start;
 let learningPreviousVolumes=[],learningContact=null;
 function renderLearning(state,openness){
  if(state.frame===0){learningPreviousVolumes=[];learningContact=null;}
  root.visible=true;
  place({nozzle:state.nozzle,eye:[0,1.32,-.1],squeeze:state.observation.state[3]/3,drop:null});
  target.visible=false;
  for(const [id,obj] of learningDrops)if(!state.drops.some(d=>d.id===id)){
   root.remove(obj);obj.geometry.dispose();obj.material.dispose();learningDrops.delete(id);
  }
  for(const d of state.drops){
   if(!learningDrops.has(d.id)){
    const obj=new THREE.Mesh(new THREE.SphereGeometry(d.radius_m,12,8),material(COLORS.drop));
    root.add(obj);learningDrops.set(d.id,obj);
   }
   learningDrops.get(d.id).position.set(...d.position);
  }
  apply({time:state.time,openness});
  const volumes=contactVolumes(parts,learningPreviousVolumes);
  learningContact=learningContact||detectContact(person,volumes);
  learningPreviousVolumes=volumes;
  return learningContact;
 }
 return {start,renderLearning,info:()=>({running,last,contact})};
}
