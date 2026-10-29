const LEGACY_TRANSLATION=.002,DEVELOPMENT_X=.001,HEAD_PITCH_STEP=.015,HEAD_YAW_STEP=.01;
const HEIGHT_LEVELS=11,HEIGHT_STEP=.01;
const RANDOM_INCREMENT=0x6D2B79F5,UINT32_RANGE=4294967296;
const RANDOM_MIX={firstShift:15,secondShift:7,finalShift:14,secondMask:61};
const MIN_HEIGHT_LEVEL=-10;
const INDEPENDENT_LEVELS={x:11,height:13,depth:11,pitch:9,yaw:7};
function independentConditions(seed,scenario){
 // Mulberry32の連続した別標本を各軸に割り当て、同一剰余による軸間の固定対応を避ける。
 let state=seed>>>0;
 const sample=levels=>{
  state=(state+RANDOM_INCREMENT)>>>0;
  let value=Math.imul(state^(state>>>RANDOM_MIX.firstShift),1|state);
  value^=value+Math.imul(value^(value>>>RANDOM_MIX.secondShift),RANDOM_MIX.secondMask|value);
  return Math.floor(((value^(value>>>RANDOM_MIX.finalShift))>>>0)/UINT32_RANGE*levels);
 };
 return {scenario,offset:[
  (sample(INDEPENDENT_LEVELS.x)-(INDEPENDENT_LEVELS.x-1)/2)*DEVELOPMENT_X,
  (sample(INDEPENDENT_LEVELS.height)+MIN_HEIGHT_LEVEL)*HEIGHT_STEP,
  (sample(INDEPENDENT_LEVELS.depth)-(INDEPENDENT_LEVELS.depth-1)/2)*LEGACY_TRANSLATION],
  pitch:(sample(INDEPENDENT_LEVELS.pitch)-(INDEPENDENT_LEVELS.pitch-1)/2)*HEAD_PITCH_STEP,
  yaw:(sample(INDEPENDENT_LEVELS.yaw)-(INDEPENDENT_LEVELS.yaw-1)/2)*HEAD_YAW_STEP};
}
export function learningConditions(seed,scenario='v6'){
 if(!Number.isSafeInteger(seed)||seed<0)throw new Error('開始条件が不正です');
 if(scenario==='people-height-v10')return independentConditions(seed,scenario);
 if(!['v6','people-v7','people-height-v9'].includes(scenario))throw new Error('未知の試行条件です');
 const development=scenario!=='v6';
 const height=scenario==='people-height-v9'?
  0-((seed*7)%HEIGHT_LEVELS)*HEIGHT_STEP:((seed*7)%9-4)*LEGACY_TRANSLATION;
 return {scenario,offset:[((seed*17)%11-5)*(development?DEVELOPMENT_X:LEGACY_TRANSLATION),height,((seed*13)%11-5)*LEGACY_TRANSLATION],pitch:development?((seed*7)%9-4)*HEAD_PITCH_STEP:0,yaw:development?((seed*13)%7-3)*HEAD_YAW_STEP:0};
}
