// 教師実演専用。学習済み方策の操作へは接続しない。
export const APPROACH=Object.freeze({
 clearance:.11,retreat:.25,retreatClearance:.015,
 originY:1.1,originZ:-.1,speed:.02,duration:15,reserve:6
});
export function teacherGoal(eye,timeBudgeted=false){
 if(eye.length!==3||!eye.every(Number.isFinite))throw new Error('教師の眼位置が不正です');
 const nominal=eye[1]+APPROACH.clearance-APPROACH.originY;
 // 待機・押圧・水平移動などのために時間を残す。成功保証ではない。
 const lower=APPROACH.retreat-APPROACH.speed*(APPROACH.duration-APPROACH.reserve)/2;
 const height=timeBudgeted?Math.max(nominal,lower):nominal;
 return [eye[0],eye[2]-APPROACH.originZ,
  Math.min(height,APPROACH.retreat-APPROACH.retreatClearance)];
}
