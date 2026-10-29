import test from 'node:test';
import assert from 'node:assert/strict';
import {teacherGoal,APPROACH} from './teacher-approach.mjs';

test('従来の教師目標を既定で維持する',()=>{
 const eye=[.04,1.1,-.18];
 assert.deepEqual(teacherGoal(eye),[eye[0],eye[2]+.1,eye[1]+.11-1.1]);
 assert.equal(teacherGoal([.04,1.25,-.18])[2],.235);
});
test('低い顔でも退避時間を残す高さから滴下する教師目標',()=>{
 const goal=teacherGoal([.04,1.0,-.18],true);
 assert.ok(Math.abs(goal[2]-.16)<1e-12);
 const verticalRoundTrip=2*(APPROACH.retreat-goal[2])/APPROACH.speed;
 assert.ok(Math.abs(verticalRoundTrip-(APPROACH.duration-APPROACH.reserve))<1e-12);
 assert.ok(goal[2]+APPROACH.originY>1.0);
});
test('不正な教師座標を黙って補完しない',()=>{
 assert.throws(()=>teacherGoal([0,NaN,0],true));
 assert.throws(()=>teacherGoal([0,1],true));
});
