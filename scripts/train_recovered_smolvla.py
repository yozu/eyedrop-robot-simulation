"""既存LeRobotとSmolVLAのLoRAで、回収した3人物実演を一候補だけ追加学習する。"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import snapshot_download

from smolvla_runtime import load_policy, ModelInput
from smolvla_training import sequence_loss
from panda_optimizer import build_optimizer, OFFICIAL
from check_smolvla_g0a import MODEL_ID, MODEL_REVISION, MEMORY_LIMIT_BYTES
from resumable_job import atomic_json

UPDATES=600
FPS=20
EPISODE_FRAMES=1200
TRAIN_EPISODES=6
VALIDATION_EPISODES=6
SEED=714
PHASES=('prepare','press','release','retreat')
GAP_PRESS_M=.0158
OPEN_MIN=.95
TIME_LIMIT_SECONDS=1800
SWAP_LIMIT_BYTES=256*1024**2
GRADIENT_LIMIT=10.
WARMUP_STEPS=60


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trainable_digest(named):
    result=hashlib.sha256()
    for name,parameter in sorted(named.items()):
        result.update(name.encode());result.update(parameter.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw');parser.add_argument('data');parser.add_argument('output')
    parser.add_argument('--stop-after',type=int,choices=(10,UPDATES),default=UPDATES)
    parser.add_argument('--resume-step',type=int,choices=(10,100,200,300,400,500));args=parser.parse_args()
    raw=Path(args.raw);data=Path(args.data);out=Path(args.output)
    if args.resume_step:
        if not out.is_dir() or (out/'result.json').exists():raise ValueError('未完了の保存学習だけを再開できます')
    else:out.mkdir(parents=True,exist_ok=False)
    gate=json.loads((data/'result.json').read_bytes());collected=json.loads((raw/'result.json').read_bytes())
    if not gate['accepted'] or gate['episodes']!={'train':TRAIN_EPISODES,'validation':VALIDATION_EPISODES} or gate['episode_frames']!=EPISODE_FRAMES:
        raise ValueError('品質確認済み3人物6/6・1200操作だけを追加学習します')
    episodes={split:[r for r in collected['episodes'] if r['split']==split] for split in ('train','validation')}
    if any(any(p in r['id'] for p in ('vroid-hair-female','vroid-hair-male')) for records in episodes.values() for r in records):
        raise ValueError('評価人物の混入を拒否します')
    groups={phase:[] for phase in PHASES};checks=[]
    for split,records in episodes.items():
        for e,record in enumerate(records):
            trace=json.loads((Path(record['original_source'])/'trace.json').read_bytes())
            if len(trace)!=EPISODE_FRAMES:raise ValueError('元の教師長が違います')
            if split=='train':
                for i,row in enumerate(trace):
                    groups[row['decision']['phase']].append(e*EPISODE_FRAMES+i)
            else:
                indices={0,EPISODE_FRAMES-1}
                for phase in PHASES:
                    ids=[i for i,row in enumerate(trace) if row['decision']['phase']==phase]
                    if not ids:raise ValueError('検査に動作段階がありません')
                    indices.update((ids[0],ids[len(ids)//2]))
                closed=[i for i,row in enumerate(trace) if row['audit_open']<OPEN_MIN]
                if not closed:raise ValueError('検査に閉眼画像がありません')
                indices.add(closed[0])
                checks.extend(dict(episode=e,step=i,index=e*EPISODE_FRAMES+i,
                    actual_open=trace[i]['audit_open'],phase=trace[i]['decision']['phase']) for i in sorted(indices))
    if any(not group for group in groups.values()):raise ValueError('学習に動作段階がありません')
    base=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,allow_patterns=['model.safetensors','config.json']))
    protocol=dict(model=MODEL_ID,revision=MODEL_REVISION,base_weights_sha256=digest(base/'model.safetensors'),updates=UPDATES,fit_calls_cap=1,
        warmup_steps=WARMUP_STEPS,scheduler_horizon=UPDATES,
        scheduler_reason='公式の1000更新warmupは600更新の有限試行を超えるため、事前に60更新warmup/600更新減衰へ固定',
        hypothesis='旧単一人物・短時間実演より、成功済み3人物の全段階を含む実演で画像条件と押圧切替を学べる',
        change='旧独自判定器を追加せず、事前学習済みSmolVLAの動作部LoRAと状態/動作射影を追加学習',
        fixed='6/6姿勢分割・RGB2枚・自己状態32・共通left-eye指示・関節7/指幅8出力・1操作再観測',
        sampler='半数を人物実演均等、半数をprepare/press/release/retreat均等。段階は教師の抽出専用で入力しない',
        validation=checks,gate='検査の関節/指幅RMSEが両方改善、閉眼誤押圧0、押圧再現数非悪化なら全動作比較。静的結果だけで採用しない',
        comparison='同じデータ/統計/指示/固定noiseで追加学習前後比較。固定制御の成功は学習成功に数えない',
        data_gate_sha256=digest(data/'result.json'),normalization_sha256=digest(data/'normalization.json'),
        source_sha256=digest(Path(__file__)),time_limit_seconds=TIME_LIMIT_SECONDS,windows_training=False)
    if args.resume_step:
        previous=json.loads((out/'protocol.json').read_bytes())
        if previous!=protocol:raise ValueError('再開時の入力・ソース・比較条件が変わっています')
    else:atomic_json(out/'protocol.json',protocol)
    train=LeRobotDataset('local/eyedrop-panda-train',root=data/'train',delta_timestamps={'action':[0.]},video_backend='pyav')
    validation=LeRobotDataset('local/eyedrop-panda-validation',root=data/'validation',delta_timestamps={'action':[0.]},video_backend='pyav')
    if len(train)!=TRAIN_EPISODES*EPISODE_FRAMES or len(validation)!=VALIDATION_EPISODES*EPISODE_FRAMES:raise ValueError('実データの件数が違います')
    resume_checkpoint=out/f'adapter-{args.resume_step}' if args.resume_step else None
    policy=load_policy(adapter=resume_checkpoint,train=True,full_projections=not bool(resume_checkpoint),chunk_size=1);inputs=ModelInput(policy,data/'normalization.json')
    tokenizer=policy.get_base_model().model.vlm_with_expert.processor.tokenizer
    instruction_tokens=tokenizer([inputs.instruction],truncation=False)['input_ids'][0]
    if len(instruction_tokens)>policy.config.tokenizer_max_length:
        raise ValueError('共通文章が長すぎるため、切り捨てて学習を進めません')
    named={n:p for n,p in policy.named_parameters() if p.requires_grad}
    for name in named:
        full='modules_to_save.default' in name and any(x in name for x in ('.state_proj.','.action_in_proj.','.action_out_proj.'))
        if not full and 'lora_' not in name:raise ValueError('許可していない重みが学習対象です')
    policy.config.scheduler_warmup_steps=WARMUP_STEPS
    policy.config.scheduler_decay_steps=UPDATES
    parameters=list(named.values());optimizer,scheduler=build_optimizer(policy.config,parameters,OFFICIAL,UPDATES)
    noise=torch.randn(1,1,32,generator=torch.Generator().manual_seed(SEED)).to('mps')
    def evaluate():
        policy.eval();rows=[]
        with torch.inference_mode():
            for check in checks:
                sample=validation[check['index']];batch=inputs.batch(sample)
                prediction=inputs.denormalize_action(policy.predict_action_chunk(batch,noise=noise)[0,0])
                target=sample['action'][0].numpy();error=prediction-target
                rows.append(dict(check,prediction=prediction.tolist(),reference=target.tolist(),error=error.tolist()))
        errors=np.array([r['error'] for r in rows]);closed=[r for r in rows if r['actual_open']<OPEN_MIN]
        presses=[r for r in rows if r['phase']=='press']
        return dict(rows=rows,joint_rmse_rad=float(np.sqrt((errors[:,:7]**2).mean())),
            gap_rmse_mm=float(np.sqrt((errors[:,7]**2).mean())*1000),
            closed_false_press=sum(r['prediction'][7]<GAP_PRESS_M for r in closed),
            press_reproduced=sum(r['prediction'][7]<GAP_PRESS_M for r in presses),press_count=len(presses))
    rng=random.Random(SEED);history=[];previous_seconds=0.;initial_trainable_sha=trainable_digest(named)
    if args.resume_step:
        saved=torch.load(resume_checkpoint/'training-state.pt',map_location='cpu',weights_only=False)
        if saved['step']!=args.resume_step:raise ValueError('保存更新数が一致しません')
        scheduler.load_state_dict(saved['scheduler']);optimizer.load_state_dict(saved['optimizer'])
        before=json.loads((out/'before-static.json').read_bytes());history=saved['history'];previous_seconds=saved['seconds']
        initial_trainable_sha=saved['initial_trainable_sha256']
        policy.eval();check_batch=inputs.batch(validation[checks[0]['index']])
        with torch.inference_mode():actual=policy.predict_action_chunk(check_batch,noise=noise).cpu()
        difference=float((actual-saved['prediction']).abs().max())
        if difference>=1e-5:raise ValueError('再開したモデルの出力が保存時と一致しません')
        rng.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.mps.set_rng_state(saved['mps_rng'])
        atomic_json(out/f'resume-{args.resume_step}.json',dict(prediction_max_abs_difference=difference,optimizer_restored=True,scheduler_restored=True,rng_restored=True))
    else:
        before=evaluate();atomic_json(out/'before-static.json',before)
    started=time.monotonic();swap=psutil.swap_memory().used
    for step in range((args.resume_step or 0)+1,args.stop_after+1):
        if step%2:
            phase=PHASES[((step-1)//2)%len(PHASES)];index=rng.choice(groups[phase])
        else:index=rng.randrange(TRAIN_EPISODES)*EPISODE_FRAMES+rng.randrange(EPISODE_FRAMES)
        policy.train();batch=inputs.batch(train[index]);optimizer.zero_grad(set_to_none=True)
        loss=sequence_loss(policy,batch,first_weight=1.,gap_weight=1.)
        if not torch.isfinite(loss):raise ValueError('学習損失が非有限です')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(parameters,GRADIENT_LIMIT)
        if not torch.isfinite(gradient):raise ValueError('学習勾配が非有限です')
        optimizer.step();scheduler.step();torch.mps.synchronize()
        rss=psutil.Process().memory_info().rss;driver=torch.mps.driver_allocated_memory()
        if rss+driver>MEMORY_LIMIT_BYTES or psutil.swap_memory().used-swap>SWAP_LIMIT_BYTES or previous_seconds+time.monotonic()-started>TIME_LIMIT_SECONDS:
            raise RuntimeError('有限の計算資源上限に達しました')
        history.append(dict(step=step,index=index,loss=float(loss.detach().cpu()),rss_bytes=rss,driver_bytes=driver))
        if step%10==0:
            atomic_json(out/'progress.json',dict(completed=False,updates=step,last=history[-1]));print(json.dumps(history[-1]),flush=True)
        if step==10 or step%100==0:
            checkpoint=out/f'adapter-{step}';policy.save_pretrained(checkpoint,safe_serialization=True)
            policy.eval();check_batch=inputs.batch(validation[checks[0]['index']])
            with torch.inference_mode():prediction=policy.predict_action_chunk(check_batch,noise=noise).cpu()
            torch.save(dict(step=step,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,
                python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),
                seconds=previous_seconds+time.monotonic()-started,prediction=prediction,
                initial_trainable_sha256=initial_trainable_sha),checkpoint/'training-state.pt')
    if args.stop_after==10:
        current_sha=trainable_digest(named)
        if current_sha==initial_trainable_sha:raise ValueError('学習対象の重みが更新されていません')
        atomic_json(out/'smoke-result.json',dict(completed=True,updates=10,training_completed=False,adopted=False,
            trainable_initial_sha256=initial_trainable_sha,trainable_after_sha256=current_sha,weights_updated=True))
        return
    after=evaluate();atomic_json(out/'after-static.json',after)
    reference_sample=validation[checks[0]['index']];batch=inputs.batch(reference_sample)
    with torch.inference_mode():reference=policy.predict_action_chunk(batch,noise=noise).cpu()
    final_trainable_sha=trainable_digest(named)
    del policy,optimizer,scheduler,parameters,named,loss,gradient;gc.collect();torch.mps.empty_cache()
    restored=load_policy(adapter=out/'adapter-600',chunk_size=1)
    with torch.inference_mode():reloaded=restored.predict_action_chunk(batch,noise=noise).cpu()
    difference=float((reference-reloaded).abs().max())
    gate=(after['joint_rmse_rad']<before['joint_rmse_rad'] and after['gap_rmse_mm']<before['gap_rmse_mm']
        and after['closed_false_press']==0 and after['press_reproduced']>=before['press_reproduced'] and difference<1e-5)
    atomic_json(out/'result.json',dict(completed=True,updates=UPDATES,static_gate=bool(gate),adopted=False,
        before={k:v for k,v in before.items() if k!='rows'},after={k:v for k,v in after.items() if k!='rows'},
        reload_max_abs_difference=difference,language_instruction=inputs.instruction,
        adapter_sha256=digest(out/'adapter-600/adapter_model.safetensors'),
        trainable_initial_sha256=initial_trainable_sha,trainable_final_sha256=final_trainable_sha,
        training_history=history,used_existing_framework=True,pretrained_action_model=True,evaluation_people_used=False))


if __name__=='__main__':main()
