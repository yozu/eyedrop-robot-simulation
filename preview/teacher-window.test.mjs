import test from 'node:test';import assert from 'node:assert/strict';
import {canStartPress,arrivalHorizon} from './teacher-window.mjs';
test('現在開眼でも到着前の閉眼を拒否',()=>assert.equal(canStartPress(0,.11,t=>t>.6&&t<.8?0:1),false));
test('開眼が続けば許可',()=>assert.equal(canStartPress(0,.11,()=>1),true));
test('閉眼中は拒否',()=>assert.equal(canStartPress(0,.11,t=>t===0?0:1),false));
test('落下距離が長いほど待機判定区間が長い',()=>assert.ok(arrivalHorizon(.3)>arrivalHorizon(.1)));
test('不正な高さを拒否',()=>assert.throws(()=>arrivalHorizon(-1)));
