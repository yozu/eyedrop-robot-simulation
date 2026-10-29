"""固定した混合教材で相対動作を学習する。対の追加損失は明示時だけ使う。"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import psutil
import torch
from accelerate import Accelerator
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.random_utils import serialize_rng_state,deserialize_rng_state

from check_smolvla_g0a import MEMORY_LIMIT_BYTES
from relative_joint_action import file_hash, bind_policy, save_policy_binding
from relative_sequence_training import UPDATES, resolve_batch, fixed_training_complete, evaluate
from resumable_job import atomic_json
from smolvla_runtime import load_policy, ModelInput, native_trainable_name
from smolvla_initial_equivalence import validate_initial_equivalence
from smolvla_training import concatenate_observations
from train_smolvla_micro_conditioning import SEED, LEARNING_RATE, WARMUP_STEPS
from sequence_training_protocol import training_protocol
from paired_switch_data import (validate_pair_preflight, validate_pair_manifest, validate_pair_resume,
                                PAIR_DIFFERENCE, PAIR_FLOW_GRID, PAIR_OBJECTIVES, FLOW_TIMES, pair_flow_time)
from paired_switch_loss import use_pair_loss
from sequence_checkpoint import (CHECKPOINT_INTERVAL, save_checkpoint, load_checkpoint, inspect_checkpoint,
                                 restore_training_state, resume_contract)

GRAD_CLIP = 10.
SWAP_LIMIT_BYTES = 256 * 1024**2
SAVE_RESERVE_SECONDS = 180
PROGRESS_INTERVAL = 10
TEACHER_CHECK_INTERVAL = 1000
PAIR_NOISE_SEED = SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('connection'); parser.add_argument('scale', choices=('old', 'relative'))
    parser.add_argument('output')
    parser.add_argument('--paired-switch-manifest', type=Path)
    parser.add_argument('--paired-switch-objective', choices=PAIR_OBJECTIVES, default=PAIR_DIFFERENCE)
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args(); connection, out = Path(args.connection), Path(args.output)
    if args.paired_switch_objective != PAIR_DIFFERENCE and not args.paired_switch_manifest:
        raise ValueError('新方式には固定した学習対の指定が必要です')
    quality = json.loads((connection / 'result.json').read_bytes())
    protocol = json.loads((connection / 'protocol.json').read_bytes())
    parent_path = Path(protocol['parent_budget']); parent = json.loads(parent_path.read_bytes())
    if not quality['ready_for_training'] or not parent['user_authorized']:
        raise ValueError('接続確認または有限予算が未成立です')
    if (datetime.fromisoformat(parent['compute_until'])-datetime.now(timezone.utc)).total_seconds()<SAVE_RESERVE_SECONDS:
        raise RuntimeError('親の計算枠に保存と再開の時間がありません')
    if args.paired_switch_manifest:
        validate_pair_preflight(parent, quality, args.paired_switch_objective)
        if args.scale != 'relative':
            raise ValueError('対の比較は観測角基準の相対関節だけを使います')
        if file_hash(args.paired_switch_manifest) != quality['paired_switch_manifest_sha256']:
            raise ValueError('接続確認後に対の一覧が変わっています')
        for name in ('paired_switch_data.py', 'paired_switch_loss.py'):
            if file_hash(Path(__file__).with_name(name)) != quality['paired_switch_source_sha256'][name]:
                raise ValueError('接続確認後に対の損失処理が変わっています')
    saved = inspect_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    decision = (resume_contract(saved,args.resume_checkpoint.parent.parent,parent,datetime.now(timezone.utc),
        source_sha256=file_hash(__file__),scale=args.scale) if saved else None)
    if not saved and parent['new_training_candidates_started'] >= parent['new_training_candidates_cap']:
        raise RuntimeError('許可済み候補上限に達しています')
    if parent['candidate_max_updates'] != UPDATES: raise ValueError('固定更新数が異なります')
    if not saved and any(row['scale'] == args.scale for row in parent.get('candidates', [])):
        raise RuntimeError('同じ尺度の候補を重複起動しません')
    now = datetime.now(timezone.utc)
    deadline = (datetime.fromisoformat(decision['deadline']) if saved else
        min(datetime.fromisoformat(parent['compute_until']), now + timedelta(seconds=parent['candidate_max_seconds'])))
    if (deadline - now).total_seconds() < SAVE_RESERVE_SECONDS: raise RuntimeError('保存の時間がありません')
    initial, rng_root, manifest = map(Path, (protocol[k] for k in ('initial','rng','manifest')))
    norm_path = connection / f'{args.scale}-normalization.json'
    if saved:
        fixed=saved['protocol']
        if (fixed['connection']!=str(connection) or fixed['parent_budget']!=str(parent_path) or
            fixed['normalization_sha256']!=file_hash(norm_path) or
            fixed['checkpoint_helper_sha256']!=file_hash(Path(__file__).with_name('sequence_checkpoint.py')) or
            fixed['official_update_source_sha256']!=hashlib.sha256(inspect.getsource(update_policy).encode()).hexdigest() or
            bool(fixed.get('paired_switch'))!=bool(args.paired_switch_manifest)):
            raise ValueError('途中保存と固定した学習条件・再開コードが異なります')
        for key in ('samples_sha256','labels_sha256','batches_sha256','helper_sha256'):
            if fixed[key]!=quality[key]:raise ValueError('再開時の教材と保存時の教材が異なります: '+key)
        for key in ('initial_model_sha256','initial_rng_sha256','manifest'):
            if fixed[key]!=protocol[key]:raise ValueError('再開時の元入力が異なります: '+key)
        if args.paired_switch_manifest and (
            fixed['paired_switch']['manifest_sha256']!=quality['paired_switch_manifest_sha256'] or
            fixed['paired_switch']['source_sha256']!=quality['paired_switch_source_sha256']):
            raise ValueError('再開時の対の学習条件が異なります')
        if args.paired_switch_manifest:
            validate_pair_resume(fixed['paired_switch'], args.paired_switch_objective)
        if (args.resume_checkpoint.parent.parent/'result.json').exists():
            raise ValueError('終了済みの候補を途中保存から再学習しません')
        saved=load_checkpoint(args.resume_checkpoint)
    for path, expected in ((connection/'samples.pt', quality['samples_sha256']),
                           (connection/'labels.json', quality['labels_sha256']),
                           (manifest/'batches.json', quality['batches_sha256']),
                           (norm_path, quality['normalization_sha256'][args.scale]),
                           (initial/'policy/model.safetensors', protocol['initial_model_sha256']),
                           (rng_root/'initial-rng.pt', protocol['initial_rng_sha256']),
                           (Path(__file__).with_name('relative_sequence_training.py'), quality['helper_sha256'])):
        if file_hash(path) != expected: raise ValueError('接続後に固定入力が変更されています: ' + str(path))
    out.mkdir(parents=True, exist_ok=False)
    cache = torch.load(connection/'samples.pt', map_location='cpu', weights_only=True)
    labels = json.loads((connection/'labels.json').read_bytes())
    pairs = (validate_pair_manifest(json.loads(args.paired_switch_manifest.read_bytes()), labels,
                                   quality['labels_sha256']) if args.paired_switch_manifest else None)
    pair_rng = torch.Generator().manual_seed(PAIR_NOISE_SEED) if pairs else None
    batches = json.loads((manifest/'batches.json').read_bytes())
    if len(batches) != UPDATES or any(len(row) != 3 for row in batches): raise ValueError('固定バッチが不正です')
    norm = json.loads(norm_path.read_bytes())
    policy = load_policy(full_policy=args.resume_checkpoint if saved else initial/'policy', native_full=True, train=True, chunk_size=1)
    bind_policy(policy, norm); inputs = ModelInput(policy, norm_path)
    named = {name:p for name,p in policy.named_parameters() if p.requires_grad}
    if not named or any(not native_trainable_name(name) for name in named): raise ValueError('動作部以外が学習対象です')
    policy.config.optimizer_lr = LEARNING_RATE
    policy.config.scheduler_warmup_steps = WARMUP_STEPS
    policy.config.scheduler_decay_steps = UPDATES
    cfg = SimpleNamespace(use_policy_training_preset=True, optimizer=policy.config.get_optimizer_preset(),
                          scheduler=policy.config.get_scheduler_preset(), steps=UPDATES)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    accelerator = Accelerator(mixed_precision='no')
    if accelerator.device.type != 'mps': raise RuntimeError('MPSが必要です')
    noise = torch.randn(1,1,32,generator=torch.Generator().manual_seed(SEED)).to('mps')
    # 9対照を再照合し、連続教材は接続時の全件結果を引き継ぐ。
    small_labels = {'train': {'micro': labels['train']['micro'], 'sequence': labels['train']['sequence'][:1]}}
    if saved:
        parity=saved['protocol']['initial_parity']
    else:
        before = evaluate(policy, inputs, cache, small_labels, 'train', noise)
        reference = json.loads((connection/f'before-{args.scale}-train.json').read_bytes())
        parity = validate_initial_equivalence(reference['micro']['rows'], before['micro']['rows'])
        for a,b in zip(reference['sequence']['rows'][0]['prediction'], before['sequence']['rows'][0]['prediction']):
            if abs(a-b)>1e-6: raise ValueError('連続教材の初期予測が不一致です')
    started_at = datetime.now(timezone.utc).isoformat()
    atomic_json(out/'protocol.json', training_protocol(protocol, scale=args.scale, connection=str(connection),
        normalization_sha256=file_hash(norm_path), samples_sha256=quality['samples_sha256'],
        labels_sha256=quality['labels_sha256'], batches_sha256=quality['batches_sha256'],
        source_sha256=file_hash(__file__), helper_sha256=quality['helper_sha256'],
        checkpoint_helper_sha256=file_hash(Path(__file__).with_name('sequence_checkpoint.py')),
        checkpoint_interval=CHECKPOINT_INTERVAL,
        official_update_source_sha256=hashlib.sha256(inspect.getsource(update_policy).encode()).hexdigest(),
        updates=UPDATES, batch_size=3, learning_rate=LEARNING_RATE, warmup=WARMUP_STEPS,
        grad_clip=GRAD_CLIP, padding_loss='official active dimensions', time_sampling='official standard',
        started_at=started_at, deadline=deadline.isoformat(), initial_parity=parity,
        validation='学習結果を確認して重み固定後に別ジョブで検査。学習の早期合格による打切りなし',
        **({'paired_switch':dict(manifest_sha256=quality['paired_switch_manifest_sha256'],
              source_sha256=quality['paired_switch_source_sha256'], noise_seed=PAIR_NOISE_SEED,
              pair_count=len(pairs), weight=1., objective=args.paired_switch_objective,
              **({'times':list(FLOW_TIMES), 'order':'pair_round_robin_then_time'}
                 if args.paired_switch_objective == PAIR_FLOW_GRID else {'time':1., 'order':'round_robin'}),
              default_rng_restored=True, inference_unchanged=True)} if pairs else {}),
        adopted=False, full_task_success=False))
    if saved:
        # 元の実行条件と期限を保持し、候補数を増やさず新しい出力先へ再開する。
        resumed_protocol={**saved['protocol'],'deadline':decision['deadline'],'checkpoint_resume':dict(
            directory=str(args.resume_checkpoint),updates=decision['offset'],
            state_sha256=file_hash(args.resume_checkpoint/'training-state.pt'),started_at=started_at)}
        atomic_json(out/'protocol.json',resumed_protocol)
    if saved:
        restore_training_state(saved['state'],optimizer,scheduler,pair_rng)
        candidate_index=decision['candidate_index']
        parent['candidates'][candidate_index].update(checkpoint_resumes_started=1,checkpoint_resume_output=str(out))
    else:
        rng = torch.load(rng_root/'initial-rng.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['cpu']); torch.mps.set_rng_state(rng['mps'])
        parent['new_training_candidates_started'] += 1
        parent.setdefault('candidates', []).append(dict(scale=args.scale, output=str(out), started_at=started_at, status='running'))
        candidate_index=len(parent['candidates'])-1
    atomic_json(parent_path, parent)
    metrics = SimpleNamespace(update_metrics=lambda values:None)
    history = list(saved['history']) if saved else []
    offset=len(history);last_checkpoint_step=offset if saved else 0
    checkpoint_protocol=json.loads((out/'protocol.json').read_bytes())
    swap = psutil.swap_memory().used; start = time.monotonic(); reason = 'fixed_updates'
    for step, descriptors in enumerate(batches[offset:], offset+1):
        if (deadline - datetime.now(timezone.utc)).total_seconds() <= SAVE_RESERVE_SECONDS:
            reason = 'deadline'; break
        batch = concatenate_observations([inputs.batch(s) for s in resolve_batch(cache['train'], descriptors)])
        pair_context = nullcontext(None)
        if pairs:
            pair = pairs[(step - 1) % len(pairs)]
            pair_descriptors = [dict(source='sequence', index=pair[key]) for key in ('before_index','after_index')]
            pair_batch = concatenate_observations([inputs.batch(s) for s in resolve_batch(cache['train'], pair_descriptors)])
            flow_time = (pair_flow_time(step, len(pairs))
                         if args.paired_switch_objective == PAIR_FLOW_GRID else 1.)
            pair_context = use_pair_loss(policy, pair_batch, pair_rng,
                                        objective=args.paired_switch_objective, flow_time=flow_time)
        with pair_context as pair_stats:
            update_policy(metrics, policy, batch, optimizer, GRAD_CLIP, accelerator, lr_scheduler=scheduler)
        if pairs and pair_stats['pair_calls'] != 1:
            raise ValueError('1更新につき対の損失を1回呼ぶ必要があります')
        torch.mps.synchronize()
        if not math.isfinite(metrics.loss) or not math.isfinite(metrics.grad_norm): raise ValueError('非有限の損失または勾配')
        rss = psutil.Process().memory_info().rss; driver = torch.mps.driver_allocated_memory()
        if rss+driver > MEMORY_LIMIT_BYTES or psutil.swap_memory().used-swap > SWAP_LIMIT_BYTES:
            raise RuntimeError('事前のメモリ上限に達しました')
        history.append(dict(step=step,loss=metrics.loss,gradient_norm=metrics.grad_norm,lr=metrics.lr,rss_bytes=rss,driver_bytes=driver))
        if pairs:
            label = 'pair_flow_loss' if args.paired_switch_objective == PAIR_FLOW_GRID else 'pair_gap_loss'
            history[-1].update(pair_episode=pair['episode'], pair_flow_time=flow_time,
                               official_loss=pair_stats['last_official_loss'],
                               **{label:pair_stats['last_pair_loss']})
        if step % PROGRESS_INTERVAL == 0:
            atomic_json(out/'progress.json',dict(completed=False,updates=step,last=history[-1]))
            print(json.dumps(history[-1]),flush=True)
        if step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(out,policy,optimizer,scheduler,norm_path,history,checkpoint_protocol,pair_rng)
            last_checkpoint_step=step
        if step % TEACHER_CHECK_INTERVAL == 0:
            # 診断が訓練乱数を消費しても両候補の抽選系列を変えない。
            check_rng=serialize_rng_state()
            try:
                check = evaluate(policy,inputs,cache,small_labels,'train',noise)
                atomic_json(out/f'train-check-{step}.json',check)
            finally:deserialize_rng_state(check_rng)
    # 終了後の全件診断が失敗しても、最後に確定した学習更新を失わない。
    if history and len(history)!=last_checkpoint_step:
        save_checkpoint(out,policy,optimizer,scheduler,norm_path,history,checkpoint_protocol,pair_rng)
    after = evaluate(policy,inputs,cache,labels,'train',noise)
    atomic_json(out/'after-train.json',after)
    policy.requires_grad_(False); fixed_at = datetime.now(timezone.utc).isoformat()
    final = out/'policy-final'; policy.save_pretrained(final)
    save_policy_binding(policy,final); (final/'normalization.json').write_bytes(norm_path.read_bytes())
    torch.save(dict(updates=len(history),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),
                    **({'pair_rng':pair_rng.get_state()} if pairs else {})),final/'training-state.pt')
    passed = all(after[source]['metrics']['passed'] for source in after)
    result = dict(completed=True, scale=args.scale,updates=len(history),fixed_updates_completed=fixed_training_complete(len(history)),
        stop_reason=reason,trainable_parameters=sum(p.numel() for p in named.values()),
        model_sha256=file_hash(final/'model.safetensors'),weights_fixed_at=fixed_at,
        after_train={k:v['metrics'] for k,v in after.items()},train_BC_passed=passed,
        seconds=time.monotonic()-start,official_update_policy=True,history=history,
        validation_run=False,adopted=False,full_task_success=False)
    atomic_json(out/'result.json',result);atomic_json(out/'progress.json',dict(completed=True,updates=len(history),result='result.json'))
    parent=json.loads(parent_path.read_bytes());parent['candidates'][candidate_index].update(status='trained_weights_fixed',updates=len(history),model_sha256=result['model_sha256'],final_output=str(out))
    atomic_json(parent_path,parent)


if __name__ == '__main__': main()
