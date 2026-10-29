import test from 'node:test';
import assert from 'node:assert/strict';
import {fixedPrediction,validateFixedActions} from './fixed-actions.mjs';

const sequence=()=>Array.from({length:300},()=>[0,0,.25,0]);
test('固定動作は300操作を先に検証し不正な列を途中まで実行しない',()=>{
 assert.doesNotThrow(()=>validateFixedActions(sequence()));
 for(const rows of [[],sequence().slice(1),sequence().concat([[0,0,.25,0]])]){
  assert.throws(()=>validateFixedActions(rows));
 }
 for(const invalid of [[0,0,.25], [NaN,0,.25,0],[0,0,.25,-.1],[0,0,.36,0]]){
  const rows=sequence();rows[299]=invalid;
  assert.throws(()=>validateFixedActions(rows));
 }
});
test('固定動作は画像や人物に依存せず指定した指令をそのまま返す',()=>{
 const rows=sequence();rows[1]=[.01,.02,.2,.3];
 const p=fixedPrediction(rows,1);
 assert.deepEqual(p.action,rows[1]);
 assert.deepEqual(p.raw_action,rows[1]);
 assert.equal(p.intervention,false);
 assert.equal(p.model_sha256,null);
 p.action[0]=.07;
 assert.equal(rows[1][0],.01);
 assert.throws(()=>fixedPrediction(rows,300));
});
