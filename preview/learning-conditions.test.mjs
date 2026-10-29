import assert from 'node:assert/strict';
import test from 'node:test';
import {learningConditions} from './learning-conditions.mjs';

test('legacy seed preserves the original displacement and pose',()=>{
 const c=learningConditions(6201,'v6');
 assert.deepEqual(c.offset,[((6201*17)%11-5)*.002,((6201*7)%9-4)*.002,((6201*13)%11-5)*.002]);
 assert.equal(c.pitch,0);assert.equal(c.yaw,0);
});
test('development teacher and evaluation use deterministic identical conditions',()=>{
 const a=learningConditions(6301,'people-v7'),b=learningConditions(6301,'people-v7');assert.deepEqual(a,b);
 assert.ok(Math.abs(a.offset[0])<=.005);assert.ok(Math.abs(a.pitch)<=.06);assert.ok(Math.abs(a.yaw)<=.03);
 assert.throws(()=>learningConditions(1,'unknown'));assert.throws(()=>learningConditions(NaN,'v6'));
});
test('高さ拡張は開発専用の別条件で、既存条件を変更しない',()=>{
 const heights=Array.from({length:40},(_,i)=>learningConditions(6201+i,'people-height-v9').offset[1]);
 assert.ok(Math.min(...heights)<=-.1);
 assert.equal(Math.max(...heights),0);
 for(let i=0;i<40;i++){
  const wide=learningConditions(6201+i,'people-height-v9');
  const old=learningConditions(6201+i,'people-v7');
  assert.equal(wide.offset[0],old.offset[0]);
  assert.equal(wide.offset[2],old.offset[2]);
  assert.equal(wide.pitch,old.pitch);
  assert.equal(wide.yaw,old.yaw);
 }
});
test('独立変動条件は同じ高さでも左右・奥行き・姿勢を複数通り含む',()=>{
 const grouped=new Map();
 for(let seed=6201;seed<6713;seed++){
  const c=learningConditions(seed,'people-height-v10');
  assert.deepEqual(c,learningConditions(seed,'people-height-v10'));
  assert.ok(Math.abs(c.offset[0])<=.005);
  assert.ok(c.offset[1]>=-.1&&c.offset[1]<=.02);
  assert.ok(Math.abs(c.offset[2])<=.01);
  assert.ok(Math.abs(c.pitch)<=.06&&Math.abs(c.yaw)<=.03);
  const group=grouped.get(c.offset[1])||[new Set(),new Set(),new Set(),new Set()];
  [c.offset[0],c.offset[2],c.pitch,c.yaw].forEach((v,i)=>group[i].add(v));
  grouped.set(c.offset[1],group);
 }
 assert.equal(grouped.size,13);
 for(const group of grouped.values())for(const values of group)assert.ok(values.size>=4);
});
