// 教師専用。簡略容器の放出待ち時間と落下時間を含めて開眼窓を確認する。
export const WINDOW=Object.freeze({releaseBound:.55,gravity:9.81,margin:.2,sample:.01,minOpen:.95});
export function arrivalHorizon(height){if(!Number.isFinite(height)||height<0)throw new Error('落下距離が不正');return WINDOW.releaseBound+Math.sqrt(2*height/WINDOW.gravity)+WINDOW.margin;}
export function canStartPress(time,height,openAt){const horizon=arrivalHorizon(height),n=Math.ceil(horizon/WINDOW.sample);for(let i=0;i<=n;i++){const value=openAt(time+i*horizon/n);if(!Number.isFinite(value)||value<WINDOW.minOpen)return false;}return true;}
