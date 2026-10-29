export const SCORING_VERSION='first-surface-open-eye-v2';
export const SPECIFIED_EYE_SCORING_VERSION='first-surface-specified-eye-v3';
export const OPEN_THRESHOLD=.9;
const DURATION_SECONDS=15;
const UV_EPSILON=1e-9;
const EYE_SIDE_EPSILON_M=1e-9;

function insidePolygon(point,polygon){
 if(!point||polygon.length<3)return false;
 const [x,y]=point;
 let inside=false;
 for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
  const [ax,ay]=polygon[j],[bx,by]=polygon[i];
  const cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax);
  if(Math.abs(cross)<=UV_EPSILON&&
   x>=Math.min(ax,bx)-UV_EPSILON&&x<=Math.max(ax,bx)+UV_EPSILON&&
   y>=Math.min(ay,by)-UV_EPSILON&&y<=Math.max(ay,by)+UV_EPSILON)return true;
  if((ay>y)!==(by>y)&&x<(bx-ax)*(y-ay)/(by-ay)+ax)inside=!inside;
 }
 return inside;
}

function matches(hit,surface){
 return surface.mesh===hit.mesh&&surface.material===hit.material&&
  (!surface.triangles||surface.triangles.includes(hit.triangle))&&
  (!surface.uv_polygon||insidePolygon(hit.uv,surface.uv_polygon));
}

// 注釈は採点だけに使用する。未注釈の人物を平面採点で成功扱いしない。
export function classifySurface(hit,openness,annotation){
 if(!annotation)return 'unscored';
 const eye=annotation.eye_surfaces.some(surface=>matches(hit,surface));
 if(eye)return openness>=OPEN_THRESHOLD?'target':'not-open';
 if(annotation.lid_regions?.some(surface=>matches(hit,surface)))return 'lid';
 if(annotation.periocular_surfaces?.some(surface=>matches(hit,surface)))return 'periocular';
 if(annotation.face_surfaces?.some(surface=>matches(hit,surface)))return 'face';
 return 'non-eye';
}

// 表面への着滴判定を保ち、両虹彩の中間面で身体の左右だけを分ける。
// この情報は採点専用で、教師の目標点や制御入力には使わない。
export function classifySpecifiedEye({hit,openness,annotation,eyes,expectedEye}){
 if(!['left','right'].includes(expectedEye))throw new Error('指定眼は身体のleftまたはrightです');
 if(!Number.isFinite(openness)||openness<0||openness>1)throw new Error('開眼状態が不正です');
 const surface=classifySurface(hit,openness,annotation);
 const result={result:surface,expected_eye:expectedEye,hit_eye:null,
  scoring_version:SPECIFIED_EYE_SCORING_VERSION};
 if(!['target','not-open'].includes(surface))return result;
 for(const point of [eyes?.left,eyes?.right,hit.point]){
  if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))
   throw new Error('指定眼の採点に必要な左右参照・着滴位置がありません');
 }
 const axis=eyes.left.map((v,i)=>v-eyes.right[i]);
 const length=Math.hypot(...axis);
 if(length<=EYE_SIDE_EPSILON_M)throw new Error('両眼の左右参照が重なっています');
 const projection=hit.point.reduce((sum,v,i)=>
  sum+(v-(eyes.left[i]+eyes.right[i])/2)*axis[i]/length,0);
 if(Math.abs(projection)<=EYE_SIDE_EPSILON_M)return {...result,result:'ambiguous-eye'};
 const side=projection>0?'left':'right';
 return {...result,hit_eye:side,result:side===expectedEye?surface:'wrong-eye'};
}

export function taskOutcome({events,dropCount,released,retreated,contact,
 simSeconds,interventions,scoringAvailable}){
 const task_success=scoringAvailable&&released&&retreated&&!contact&&
  simSeconds<=DURATION_SECONDS&&dropCount===1&&events.length===1&&
  events[0].result==='target';
 return {task_success,success:task_success&&interventions===0};
}
