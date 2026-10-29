import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {classifySurface} from './anatomy-score.mjs';

const annotations=JSON.parse(fs.readFileSync(
 new URL('./assets/anatomy-annotations.json',import.meta.url)));
for(const person of ['alicia','tsukuyomi']){
 test(`${person}: 撮影画像で確認した眼と閉眼面を区別する`,()=>{
  const picks=JSON.parse(fs.readFileSync(new URL(
   `../../evidence/development/heldout-preparation/${person}-picks.json`,import.meta.url)));
  const annotation=annotations.models[person];
  for(const eye of ['leftIris','rightIris']){
   assert.equal(classifySurface(picks.open[eye],1,annotation),'target');
   assert.notEqual(classifySurface(picks.closed[eye],0,annotation),'target');
  }
  for(const part of ['skin','leftWhite']){
   assert.notEqual(classifySurface(picks.open[part],1,annotation),'target');
  }
 });
}
