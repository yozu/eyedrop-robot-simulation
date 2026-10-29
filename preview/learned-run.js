import {taskOutcome} from './anatomy-score.mjs';
import {fixedPrediction,validateFixedActions} from './fixed-actions.mjs';
const HZ=20,DURATION_SECONDS=15,MAX_FRAMES=HZ*DURATION_SECONDS,TARGET_RADIUS=.006;
const HISTORY_STEPS=8;
const DEFAULT_TRIAL=101;
const RELEASE_TOLERANCE_MM=.05;
const RETREAT_HEIGHT_M=.25,RETREAT_TOLERANCE_M=.005;
const MAX_INFERENCE_AGE_MS=2000;
async function request(path,body){
 const r=await fetch(`/api/learning/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok)throw new Error(await r.text());return r.json();
}

export function mountLearningRunner(view){
 const panel=document.querySelector('aside');
 const button=document.createElement('button');button.textContent='学習済みACTで動かす';
 const trial=Number(new URLSearchParams(location.search).get('trial')||DEFAULT_TRIAL);
 const pressureMemory=new URLSearchParams(location.search).get('pressure-memory')==='1';
 if(!Number.isSafeInteger(trial)||trial<0)throw new Error('開始条件が不正です');
 const status=document.createElement('p');status.textContent=`画像＋自身の動作履歴から操作します。開始条件${trial}。学習に使った条件かどうかは、モデルごとの評価記録に記載しています。`;
 panel.append(document.createElement('hr'),button,status);
 let running=false;
 async function run(seed=DEFAULT_TRIAL,fixedActions=null){
  if(running)throw new Error('学習済み方策を実行中です');
  if(fixedActions!==null)validateFixedActions(fixedActions);
  running=true;button.disabled=true;
  const log={seed,scenario:view.scenario,pressure_memory:pressureMemory,mode:fixedActions?'fixed-motion':'learned-act',frames:[],scores:[],drop_events:[],score_events:[],
   surface_events:[],surface_scoring:view.scoring,scoring_available:view.scoring.available,
   contact:null,success:false};
  const started=performance.now();
  window.learnedRunResult=null;
  try{
   view.reset(seed);
   let sim=await request('reset',{seed}),runId=sim.run;
   const history=Array.from({length:HISTORY_STEPS},()=>sim.observation.state.slice());
   const scored=new Set();
   const surfaceScored=new Set();
   for(let frame=0;frame<MAX_FRAMES;frame++){
    const rendered=view.render(sim);
    if(rendered.contact){log.contact=rendered.contact;break;}
    // 方策には正解座標・人物ID・時刻・開眼量・教師指令を渡さない。
    let prediction;
    if(fixedActions){
     prediction=fixedPrediction(fixedActions,frame);
    }else{
     const image=view.capture().split(',')[1],observedAt=performance.now();
     prediction=await request('infer',{run:runId,frame:sim.frame,image,state:sim.observation.state,history});
     if(performance.now()-observedAt>MAX_INFERENCE_AGE_MS)throw new Error('推論応答が古いため停止しました');
     if(Boolean(prediction.pressure_memory)!==pressureMemory)throw new Error('モデルと押圧履歴の設定が一致しません');
    }
    const next=await request('step',{run:runId,frame:sim.frame,action:prediction.action});
    for(const drop of next.drops){
     if(!sim.drops.some(previous=>previous.id===drop.id)){
      log.drop_events.push({id:drop.id,time:next.time,state:next.observation.state});
     }
    }
    // 以下の正解情報は独立した採点だけに使用する。方策の操作を書き換えない。
    view.render(next);
    const truth=view.teacher();
    log.plane_scoring_available=truth.eye!==null;
    for(const d of next.drops){
     const old=sim.drops.find(x=>x.id===d.id);
     const previous=old?.position||sim.nozzle;
     // 最初に触れた実形状を別記録する。材質名だけで眼・まぶたの領域を断定しない。
     if(!surfaceScored.has(d.id)){
      const hit=view.surfaceHit(previous,d.position);
      if(hit){
       log.surface_events.push({id:d.id,time:next.time,...hit,openness:truth.openness,
        result:view.classifyHit(hit,truth.openness)});
       surfaceScored.add(d.id);
      }
     }
     if(scored.has(d.id))continue;
     if(truth.eye&&previous[1]>truth.eye[1]&&d.position[1]<=truth.eye[1]){
      const fraction=(previous[1]-truth.eye[1])/(previous[1]-d.position[1]);
      const x=previous[0]+fraction*(d.position[0]-previous[0])-truth.eye[0];
      const z=previous[2]+fraction*(d.position[2]-previous[2])-truth.eye[2];
      log.scores.push(x*x+z*z<=TARGET_RADIUS**2?(truth.openness>=.9?'target':'lid'):'face');
      log.score_events.push({id:d.id,time:next.time,openness:truth.openness,
       distance:Math.hypot(x,z),result:log.scores.at(-1)});
      scored.add(d.id);
     }
    }
    log.frames.push({state:sim.observation.state,action:prediction.action,
     raw_action:prediction.raw_action,intervention:prediction.intervention,service_ms:prediction.service_ms,
     predicted_face_features:prediction.predicted_face_features||null});
    if(log.model_sha256&&log.model_sha256!==prediction.model_sha256)throw new Error('実行中に操作モデルが変わりました');
    if(log.face_model_sha256&&log.face_model_sha256!==prediction.face_model_sha256)throw new Error('実行中に顔モデルが変わりました');
    log.model_sha256=prediction.model_sha256;log.face_model_sha256=prediction.face_model_sha256||null;sim=next;
    if(!pressureMemory)history.shift();
    history.push(sim.observation.state.slice());
    status.textContent=`ACT実行中：${frame+1}/${MAX_FRAMES} ／ ${sim.drops.length}滴`;
   }
   const finalRender=view.render(sim);
   log.contact=log.contact||finalRender.contact;
   log.drop_count=sim.drops.length;
   log.sim_seconds=sim.time;
   log.final_state=sim.observation.state;
   log.released=sim.observation.state[3]<=RELEASE_TOLERANCE_MM;
   log.retreated=Math.abs(sim.observation.state[2]-RETREAT_HEIGHT_M)<=RETREAT_TOLERANCE_M;
   log.plane_task_success=log.released&&log.retreated&&!log.contact&&sim.drops.length===1&&log.scores.length===1&&log.scores[0]==='target';
   const interventions=log.frames.filter(frame=>frame.intervention).length;
   log.surface_scores=log.surface_events.map(event=>event.result);
   Object.assign(log,taskOutcome({events:log.surface_events,dropCount:log.drop_count,
    released:log.released,retreated:log.retreated,contact:log.contact,simSeconds:log.sim_seconds,
    interventions,scoringAvailable:log.scoring_available}));
   status.textContent=`ACT結果：${log.drop_count}滴 ／ ${log.contact?'接触停止':log.surface_scores.join('・')||'着滴なし'} ／ 最終押圧${sim.observation.state[3].toFixed(3)}mm ／ 範囲補正${interventions}回（${log.scoring_available?'形状による採点':'人物の採点注釈が未設定'}）`;
  }catch(error){log.error=error.message;status.textContent=`ACT停止：${error.message}`;throw error;
  }finally{log.wall_seconds=(performance.now()-started)/1000;running=false;button.disabled=false;window.learnedRunResult=log;}
  return log;
 }
 button.onclick=()=>run(trial).catch(()=>{});
 window.runLearnedPolicy=run;
}
