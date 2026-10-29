import test from 'node:test';
import assert from 'node:assert/strict';
import {classifySpecifiedEye, classifySurface, SPECIFIED_EYE_SCORING_VERSION} from './anatomy-score.mjs';

const annotation={eye_surfaces:[{mesh:'eyes',material:'iris'}]};
const eyes={left:[.03,1.2,.1],right:[-.03,1.2,.1]};
const iris=point=>({mesh:'eyes',material:'iris',point});
const score=(hit,openness=1,centers=eyes,expectedEye='left')=>
 classifySpecifiedEye({hit,openness,annotation,eyes:centers,expectedEye});

test('同じ虹彩材質でも指定した身体左眼だけを成功にする',()=>{
 assert.equal(score(iris(eyes.left)).result,'target');
 assert.equal(score(iris(eyes.right)).result,'wrong-eye');
 assert.equal(score(iris(eyes.right)).hit_eye,'right');
 assert.equal(score(iris(eyes.left)).expected_eye,'left');
 assert.equal(score(iris(eyes.left)).scoring_version,SPECIFIED_EYE_SCORING_VERSION);
});

test('世界座標の正負を左右と決めず、移動と反転に追従する',()=>{
 const moved={left:[-.63,2.2,.7],right:[-.57,2.2,.7]};
 assert.equal(score(iris(moved.left),1,moved).result,'target');
 assert.equal(score(iris(moved.right),1,moved).result,'wrong-eye');
 assert.equal(score(iris(moved.right),1,moved,'right').result,'target');
});

test('指定眼でも閉眼・半開眼・皮膚・未注釈を成功にしない',()=>{
 for(const open of [0,.5])assert.equal(score(iris(eyes.left),open).result,'not-open');
 assert.equal(score({mesh:'face',material:'skin',point:eyes.left}).result,'non-eye');
 assert.equal(classifySpecifiedEye({hit:iris(eyes.left),openness:1,annotation:null,
  eyes:null,expectedEye:'left'}).result,'unscored');
});

test('左右の境界や不正な左右参照を対象眼へ決め打ちしない',()=>{
 assert.equal(score(iris([0,1.2,.1])).result,'ambiguous-eye');
 assert.throws(()=>score(iris(eyes.left),1,{left:eyes.left,right:eyes.left}));
 assert.throws(()=>score(iris(eyes.left),1,{left:[NaN,0,0],right:eyes.right}));
 assert.throws(()=>score(iris(eyes.left),1,eyes,'both'));
 assert.throws(()=>score({mesh:'eyes',material:'iris'}));
});

test('保存済みの旧採点を変更せず、新採点と識別できる',()=>{
 assert.equal(classifySurface(iris(eyes.right),1,annotation),'target');
 assert.notEqual(SPECIFIED_EYE_SCORING_VERSION,'first-surface-open-eye-v2');
});
