const FRAMES=300,AXES=4;
const LOW=[-.08,-.16,.02,0],HIGH=[.08,.12,.35,3];

export function validateFixedActions(actions){
 if(!Array.isArray(actions)||actions.length!==FRAMES)
  throw new Error('固定動作は300操作で指定してください');
 for(const row of actions){
  if(!Array.isArray(row)||row.length!==AXES||
   row.some((value,axis)=>!Number.isFinite(value)||value<LOW[axis]||value>HIGH[axis]))
   throw new Error('固定動作の機構指令が不正です');
 }
}

// 評価人物の座標や画像で軌道を調整せず、事前固定した指令だけを再生する。
export function fixedPrediction(actions,frame){
 if(!Number.isInteger(frame)||frame<0||frame>=actions.length)
  throw new Error('固定動作の操作番号が不正です');
 return {action:actions[frame].slice(),raw_action:actions[frame].slice(),
  intervention:false,service_ms:0,model_sha256:null};
}
