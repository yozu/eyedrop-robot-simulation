"""同じ保存入力で、対の損失を実SmolVLAへ接続し0更新で検証する。"""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time

import psutil
import torch

from budget_cycle import experiment_timeout
from check_smolvla_g0a import MEMORY_LIMIT_BYTES
from paired_switch_data import (validate_pair_manifest, validate_pair_authorization,
                                PAIR_DIFFERENCE, PAIR_FLOW_GRID, PAIR_OBJECTIVES, FLOW_TIMES)
from paired_switch_loss import paired_flow_loss, use_pair_loss
from relative_joint_action import bind_policy, file_hash, save_policy_binding
from relative_sequence_training import resolve_batch, evaluate
from resumable_job import atomic_json
from smolvla_initial_equivalence import validate_initial_equivalence
from smolvla_runtime import load_policy, ModelInput
from smolvla_training import concatenate_observations

SEED = 714
ROUNDTRIP_LIMIT = 1e-6
SWAP_LIMIT_BYTES = 256 * 1024**2
PAIR_SOURCES = ('paired_switch_data.py', 'paired_switch_loss.py')


def main():
    parser = argparse.ArgumentParser()
    for name in ('proposal', 'budget', 'manifest', 'output'): parser.add_argument(name)
    parser.add_argument('--paired-switch-objective', choices=PAIR_OBJECTIVES, default=PAIR_DIFFERENCE)
    args = parser.parse_args()
    parent = json.loads(Path(args.budget).read_bytes())
    if not parent['user_authorized'] or not parent['paired_switch_authorized']:
        raise ValueError('対の学習の有限枠が必要です')
    validate_pair_authorization(parent, args.paired_switch_objective)
    experiment_timeout(parent, datetime.now(timezone.utc), 600)
    proposal = json.loads(Path(args.proposal).read_bytes()); fixed = proposal['fixed']
    if proposal.get('paired_switch_objective', PAIR_DIFFERENCE) != args.paired_switch_objective:
        raise ValueError('事前登録と対の学習方式が異なります')
    old, out = Path(fixed['connection']), Path(args.output)
    quality = json.loads((old/'result.json').read_bytes())
    protocol = json.loads((old/'protocol.json').read_bytes())
    checks = [(old/'samples.pt', fixed['samples_sha256']),
              (old/'labels.json', fixed['labels_sha256']),
              (old/'relative-normalization.json', fixed['normalization_sha256']),
              (Path(protocol['manifest'])/'batches.json', fixed['batches_sha256']),
              (Path(protocol['initial'])/'policy/model.safetensors', fixed['initial_model_sha256']),
              (Path(protocol['rng'])/'initial-rng.pt', fixed['initial_rng_sha256'])]
    for path, expected in checks:
        if file_hash(path) != expected: raise ValueError('固定入力の変更: '+str(path))
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); swap_start = psutil.swap_memory().used
    # 大きなキャッシュは読取専用で再利用し、旧成果の内容は更新しない。
    os.link(old/'samples.pt', out/'samples.pt')
    for name in ('labels.json', 'relative-normalization.json', 'before-relative-train.json'):
        (out/name).write_bytes((old/name).read_bytes())
    atomic_json(out/'protocol.json', {**protocol, 'parent_budget':args.budget,
                                    'paired_switch_objective':args.paired_switch_objective,
                                    'paired_proposal_sha256':file_hash(args.proposal)})
    labels = json.loads((out/'labels.json').read_bytes())
    pairs = validate_pair_manifest(json.loads(Path(args.manifest).read_bytes()), labels, fixed['labels_sha256'])
    cache = torch.load(out/'samples.pt', map_location='cpu', weights_only=True)
    batches = json.loads((Path(protocol['manifest'])/'batches.json').read_bytes())
    norm_path = out/'relative-normalization.json'; norm = json.loads(norm_path.read_bytes())
    policy = load_policy(full_policy=Path(protocol['initial'])/'policy', native_full=True, train=True, chunk_size=1)
    bind_policy(policy, norm); inputs = ModelInput(policy, norm_path)
    batch = concatenate_observations([inputs.batch(s) for s in resolve_batch(cache['train'], batches[0])])
    descriptors = [dict(source='sequence', index=pairs[0][key]) for key in ('before_index','after_index')]
    pair_batch = concatenate_observations([inputs.batch(s) for s in resolve_batch(cache['train'], descriptors)])
    names = {n:p for n,p in policy.named_parameters() if p.requires_grad}
    cpu, mps = torch.get_rng_state(), torch.mps.get_rng_state()
    loss, _ = policy.forward(batch); loss.backward(); torch.mps.synchronize()
    original_loss = float(loss.detach())
    gradients = {n:p.grad.detach().cpu().clone() for n,p in names.items() if p.grad is not None}
    prefixes = ('model.state_proj.', 'model.vlm_with_expert.lm_expert.', 'model.action_out_proj.')
    official_groups = {prefix:sum(float(g.square().sum()) for n,g in gradients.items()
                                 if n.startswith(prefix))**.5 for prefix in prefixes}
    del loss
    policy.zero_grad(set_to_none=True); torch.set_rng_state(cpu); torch.mps.set_rng_state(mps)
    generator = torch.Generator().manual_seed(SEED); pair_rng = generator.get_state().clone()
    with use_pair_loss(policy, pair_batch, generator, weight=0,
                       objective=args.paired_switch_objective, flow_time=1.):
        loss, _ = policy.forward(batch); loss.backward()
    torch.mps.synchronize()
    loss_difference = abs(float(loss.detach())-original_loss)
    grad_names = {n for n,p in names.items() if p.grad is not None}
    if grad_names != set(gradients): raise ValueError('係数0で勾配対象が変化しました')
    gradient_difference = max(float((p.grad.detach().cpu()-gradients[n]).abs().max())
                              for n,p in names.items() if n in gradients)
    if loss_difference != 0 or gradient_difference != 0 or not torch.equal(pair_rng, generator.get_state()):
        raise ValueError('係数0で公式損失・勾配・対の乱数が変化しました')
    del gradients, loss
    policy.zero_grad(set_to_none=True)
    checks = []
    times = FLOW_TIMES if args.paired_switch_objective == PAIR_FLOW_GRID else (1.,)
    for flow_time in times:
        before_cpu, before_mps = torch.get_rng_state(), torch.mps.get_rng_state()
        pair_loss = paired_flow_loss(policy, pair_batch, generator,
                                    objective=args.paired_switch_objective, flow_time=flow_time)
        rng_preserved = torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_mps,torch.mps.get_rng_state())
        pair_loss.backward(); torch.mps.synchronize()
        finite = all(torch.isfinite(p.grad).all().item() for p in names.values() if p.grad is not None)
        groups = {}
        for prefix in prefixes:
            groups[prefix] = sum(float(p.grad.detach().square().sum()) for n,p in names.items()
                                 if n.startswith(prefix) and p.grad is not None)**.5
        if (not finite or not torch.isfinite(pair_loss) or not rng_preserved or
                any(not math.isfinite(value) or value <= 0 for value in groups.values())):
            raise ValueError('対の損失の有限勾配・対象・乱数保持が不成立です')
        pair_value = float(pair_loss.detach()); del pair_loss
        checks.append(dict(time=flow_time, loss=pair_value, gradient_norms=groups,
                           default_rng_preserved=rng_preserved))
        policy.zero_grad(set_to_none=True)
    with use_pair_loss(policy, pair_batch, generator,
                       objective=args.paired_switch_objective, flow_time=times[-1]) as stats:
        combined, _ = policy.forward(batch); combined.backward()
    torch.mps.synchronize()
    if stats['pair_calls'] != 1 or not torch.isfinite(combined) or not all(
            torch.isfinite(p.grad).all().item() for p in names.values() if p.grad is not None):
        raise ValueError('実モデルの合成損失・勾配が不成立です')
    del combined
    policy.zero_grad(set_to_none=True)
    resources = dict(rss_bytes=psutil.Process().memory_info().rss,
                     driver_bytes=torch.mps.driver_allocated_memory(),
                     swap_delta_bytes=psutil.swap_memory().used-swap_start)
    if resources['rss_bytes']+resources['driver_bytes'] > MEMORY_LIMIT_BYTES or resources['swap_delta_bytes'] > SWAP_LIMIT_BYTES:
        raise RuntimeError('接続試験でメモリ上限に達しました')
    small = {'train':{'micro':labels['train']['micro'], 'sequence':labels['train']['sequence'][:1]}}
    noise = torch.randn(1,1,32,generator=torch.Generator().manual_seed(SEED)).to('mps')
    before = evaluate(policy, inputs, cache, small, 'train', noise)
    reference = json.loads((out/'before-relative-train.json').read_bytes())
    parity = validate_initial_equivalence(reference['micro']['rows'], before['micro']['rows'])
    if max(abs(a-b) for a,b in zip(reference['sequence']['rows'][0]['prediction'],before['sequence']['rows'][0]['prediction'])) > ROUNDTRIP_LIMIT:
        raise ValueError('連続例の初期値が一致しません')
    saved = out/'roundtrip-policy'; policy.save_pretrained(saved); save_policy_binding(policy,saved)
    (saved/'normalization.json').write_bytes(norm_path.read_bytes())
    del names, policy, inputs, batch, pair_batch; gc.collect(); torch.mps.empty_cache()
    restored = load_policy(full_policy=saved, native_full=True, chunk_size=1)
    restored_inputs = ModelInput(restored,norm_path)
    after = evaluate(restored,restored_inputs,cache,small,'train',noise)
    maximum = max(abs(a-b) for source in ('micro','sequence')
                  for r,s in zip(before[source]['rows'],after[source]['rows'])
                  for a,b in zip(r['prediction'],s['prediction']))
    if maximum > ROUNDTRIP_LIMIT: raise ValueError('保存再読込で予測が一致しません')
    atomic_json(out/'roundtrip-predictions.json',dict(before=before,after=after))
    atomic_json(out/'result.json', {**quality, 'completed':True, 'updates':0,
        'ready_for_training':True, 'paired_switch_ready':True,
        'paired_switch_objective':args.paired_switch_objective, 'pair_time_checks':checks,
        'paired_switch_manifest_sha256':file_hash(args.manifest),
        'paired_switch_source_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in PAIR_SOURCES},
        'source_sha256':file_hash(__file__), 'zero_weight_loss_difference':loss_difference,
        'zero_weight_gradient_difference':gradient_difference, 'gradient_parameters_checked':len(grad_names),
        'pair_loss':pair_value, 'pair_gradient_norms':groups, 'default_rng_preserved':rng_preserved,
        'official_gradient_norms':official_groups,
        'combined_loss':stats, 'initial_parity':parity, 'roundtrip_max':maximum,
        'resources':resources, 'seconds':time.monotonic()-start,
        'scope':'実SmolVLAの0更新接続確認。学習性能・全動作は未確認'})


if __name__ == '__main__': main()
