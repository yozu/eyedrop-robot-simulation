import test from 'node:test';
import assert from 'node:assert/strict';
import {classifySurface,taskOutcome} from './anatomy-score.mjs';

const annotation={eye_surfaces:[{mesh:'eye',material:'iris'}]};
const eye={mesh:'eye',material:'iris'};
const passing={events:[{result:'target'}],dropCount:1,released:true,retreated:true,
 contact:null,simSeconds:15,interventions:0,scoringAvailable:true};
test('注釈された眼の形状に開眼時に当たった場合だけ対象への着滴',()=>{
 assert.equal(classifySurface(eye,1,annotation),'target');
 assert.equal(classifySurface(eye,.5,annotation),'not-open');
 assert.equal(classifySurface({mesh:'face',material:'iris'},1,annotation),'non-eye');
 assert.equal(classifySurface(eye,1,null),'unscored');
});
test('まばたきで動く周辺面をまぶたや通常の顔に断定しない',()=>{
 const labels={eye_surfaces:[],
  lid_regions:[{mesh:'head',material:'skin',triangles:[1]}],
  periocular_surfaces:[{mesh:'head',material:'skin',triangles:[1,2]}],
  face_surfaces:[{mesh:'head',material:'skin'}]};
 const hit={mesh:'head',material:'skin'};
 assert.equal(classifySurface({...hit,triangle:1},0,labels),'lid');
 assert.equal(classifySurface({...hit,triangle:2},0,labels),'periocular');
 assert.equal(classifySurface({...hit,triangle:3},0,labels),'face');
});
test('未注釈や閉眼は成功せず、平面採点で代用しない',()=>{
 assert.equal(taskOutcome(passing).task_success,true);
 for(const result of ['not-open','non-eye','unscored']){
  assert.equal(taskOutcome({...passing,events:[{result}]}).task_success,false);
 }
 assert.equal(taskOutcome({...passing,scoringAvailable:false}).task_success,false);
});
test('一滴・解除・退避・非接触・時間を全て満たし補正を別集計',()=>{
 for(const change of [{dropCount:2},{released:false},{retreated:false},{contact:{}},
  {simSeconds:15.05},{events:[]}]){
  assert.equal(taskOutcome({...passing,...change}).task_success,false);
 }
 assert.deepEqual(taskOutcome({...passing,interventions:1}),
  {task_success:true,success:false});
});
test('同じ材質の虹彩とまぶたを三角形と領域で分ける',()=>{
 const labels={eye_surfaces:[{mesh:'head',material:'eye',triangles:[1]}],
  lid_regions:[{mesh:'head',material:'eye',triangles:[2],
   uv_polygon:[[.2,.2],[.4,.2],[.4,.4],[.2,.4]]}],
  face_surfaces:[{mesh:'head',material:'eye'}]};
 const point={mesh:'head',material:'eye',triangle:2,uv:[.3,.3]};
 assert.equal(classifySurface({...point,triangle:1},1,labels),'target');
 assert.equal(classifySurface(point,0,labels),'lid');
 assert.equal(classifySurface({...point,uv:[.2,.2]},0,labels),'lid');
 assert.equal(classifySurface({...point,uv:[.8,.8]},0,labels),'face');
 assert.equal(classifySurface({...point,triangle:3},0,labels),'face');
 assert.equal(classifySurface({...point,mesh:'clothes'},0,labels),'non-eye');
});
