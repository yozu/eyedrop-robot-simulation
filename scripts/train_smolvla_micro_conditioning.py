"""許可済みの一候補だけで、少数の条件付き単一操作が学習できるか確認する。"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from resumable_job import atomic_json

MAX_UPDATES = 1000
BATCH_SIZE = 3
WARMUP_STEPS = 50
LEARNING_RATE = 1e-4
SEED = 714
OPEN_MINIMUM = .95
PRESS_GAP_M = .0158
JOINT_RMSE_LIMIT_RAD = .001
JOINT_MAX_LIMIT_RAD = .005
GAP_MAX_LIMIT_M = .00025
CHECK_STEPS = (100, 300, 600, MAX_UPDATES)
FINAL_EVALUATION_RESERVE_S = 90
GRADIENT_LIMIT = 10.
SWAP_LIMIT_BYTES = 256 * 1024**2
OBSERVATION_KEYS = ('observation.state', 'observation.images.camera1', 'observation.images.camera2')
STANDARD_SAMPLING = 'standard'
ENDPOINT_SAMPLING = 'half-endpoint'
ENDPOINT_INTERVAL = 2
INFERENCE_START_TIME = 1.


def sample_training_conditions(flow, action_shape, device, step, schedule):
    import torch
    if type(step) is not int or step < 1 or schedule not in (STANDARD_SAMPLING, ENDPOINT_SAMPLING):
        raise ValueError('学習時刻の指定が不正です')
    # 公式forwardと同じ順序・個数を抽選し、時刻を置換しても乱数系列を保つ。
    noise = flow.sample_noise(action_shape, device)
    times = flow.sample_time(action_shape[0], device)
    if schedule == ENDPOINT_SAMPLING and step % ENDPOINT_INTERVAL == 0:
        times = torch.full_like(times, INFERENCE_START_TIME)
    return noise, times


def model_observation(sample):
    return {key: sample[key] for key in OBSERVATION_KEYS}


def check_metrics(rows):
    errors = np.asarray([np.asarray(row['prediction']) - row['reference'] for row in rows])
    opened = [row for row in rows if row['actual_open'] >= OPEN_MINIMUM]
    closed = [row for row in rows if row['actual_open'] < OPEN_MINIMUM]
    joint_rmse = float(np.sqrt(np.mean(errors[:, :7] ** 2)))
    joint_max = float(np.max(np.abs(errors[:, :7])))
    gap_max = float(np.max(np.abs(errors[:, 7])))
    presses = sum(row['prediction'][7] < PRESS_GAP_M for row in opened)
    releases = sum(abs(row['prediction'][7] - row['reference'][7]) <= GAP_MAX_LIMIT_M for row in closed)
    passed = bool(np.isfinite(errors).all() and opened and closed and
                  joint_rmse <= JOINT_RMSE_LIMIT_RAD and joint_max <= JOINT_MAX_LIMIT_RAD and
                  gap_max <= GAP_MAX_LIMIT_M and presses == len(opened) and releases == len(closed))
    return dict(passed=passed, joint_rmse_rad=joint_rmse, joint_max_abs_rad=joint_max,
                gap_max_abs_mm=gap_max * 1000, open_press=presses, open_count=len(opened),
                closed_half_release=releases, closed_half_count=len(closed))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import psutil
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from check_smolvla_g0a import MEMORY_LIMIT_BYTES, MODEL_ID, MODEL_REVISION
    from panda_optimizer import build_optimizer, OFFICIAL
    from smolvla_runtime import load_policy, ModelInput
    from smolvla_training import concatenate_observations, sequence_loss
    from train_recovered_smolvla import trainable_digest

    parser = argparse.ArgumentParser()
    for name in ('data', 'initial_adapter', 'policy', 'output'):
        parser.add_argument(name)
    parser.add_argument('--time-sampling', choices=(STANDARD_SAMPLING, ENDPOINT_SAMPLING), default=STANDARD_SAMPLING)
    args = parser.parse_args()
    data, initial, policy_path, out = map(Path, (args.data, args.initial_adapter, args.policy, args.output))
    budget = json.loads(policy_path.read_bytes())
    sampling = args.time_sampling
    if budget.get('time_sampling', STANDARD_SAMPLING) != sampling:
        raise ValueError('計算枠に登録した学習時刻と異なります')
    deadline = datetime.fromisoformat(budget['compute_until'])
    if not budget['user_authorized'] or budget['training_candidates_cap'] != 1:
        raise ValueError('新しい一候補の実行許可が必要です')
    if (deadline - datetime.now(timezone.utc)).total_seconds() <= FINAL_EVALUATION_RESERVE_S:
        raise RuntimeError('検査と保存の時間が残っていません')
    quality = json.loads((data / 'result.json').read_bytes())
    if not quality['data_quality_accepted'] or quality['episodes'] != {'train': 9, 'validation': 9}:
        raise ValueError('品質確認済み9/9データが必要です')
    if digest(initial / 'adapter_model.safetensors') != budget['initial_adapter_sha256']:
        raise ValueError('開始重みが許可した候補と異なります')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'protocol.json', dict(
        model=MODEL_ID, model_revision=MODEL_REVISION, user_authorized=True,
        hypothesis=('同じ自己状態で開閉RGBを対照化し、少数実演に反復提示すれば動作切替を学べるか' if sampling == STANDARD_SAMPLING else
                    '推論開始の純粋ノイズ時刻を半分の更新で提示すると、RGBに依存する押圧/解除の分離を学べるか'),
        refutation='1000更新または20分で学習9例自体を再現できなければ人物/全動作を増やさず学習経路を再点検',
        updates_cap=MAX_UPDATES, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        warmup=WARMUP_STEPS, check_steps=CHECK_STEPS, validation_selection='学習側の最初の合格または上限で固定後、検査を前後各1回。検査でcheckpoint選別しない',
        time_sampling=sampling, endpoint_interval=ENDPOINT_INTERVAL, inference_start_time=INFERENCE_START_TIME,
        comparison='同じ旧600重み・9/9画像・batch3・乱数seedと抽選順・1000更新上限。通常時刻分布の実行済み候補と比較',
        source_sha256=digest(__file__), budget_sha256=digest(policy_path),
        initial_adapter_sha256=digest(initial / 'adapter_model.safetensors'),
        normalization_sha256=digest(data / 'normalization.json'), data_quality_sha256=digest(data / 'result.json'),
        compute_until=deadline.isoformat(), adopted=False,
        inference='RGB2枚・自己状態32・保存共通文章。条件や正解は入力しない',
        teacher_scope='単一操作の合成対照。全動作成功ではない'))
    datasets = {split: LeRobotDataset(f'local/eyedrop-micro-{split}', root=data / 'dataset' / split,
                   delta_timestamps={'action': [0.]}, video_backend='pyav') for split in ('train', 'validation')}
    labels = {split: json.loads((data / f'{split}-frames.json').read_bytes()) for split in datasets}
    for split, ds in datasets.items():
        if len(ds) != 9:
            raise ValueError('事前固定した9例と違います')
        for i, label in enumerate(labels[split]):
            np.testing.assert_array_equal(ds[i]['action'][0].numpy(), np.asarray(label['action'], dtype=np.float32))
    policy = load_policy(adapter=initial, train=True, chunk_size=1)
    inputs = ModelInput(policy, data / 'normalization.json')
    named = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    if not named:
        raise ValueError('学習する重みがありません')
    initial_trainable = trainable_digest(named)
    policy.config.optimizer_lr = LEARNING_RATE
    policy.config.scheduler_warmup_steps = WARMUP_STEPS
    policy.config.scheduler_decay_steps = MAX_UPDATES
    optimizer, scheduler = build_optimizer(policy.config, list(named.values()), OFFICIAL, MAX_UPDATES)
    noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(SEED)).to('mps')

    def evaluate(split):
        policy.eval()
        rows = []
        with torch.inference_mode():
            for i, label in enumerate(labels[split]):
                sample = datasets[split][i]
                prediction = inputs.denormalize_action(policy.predict_action_chunk(inputs.batch(model_observation(sample)), noise=noise)[0, 0])
                rows.append(dict(index=i, actual_open=label['actual_open'], prediction=prediction.tolist(),
                                 reference=sample['action'][0].numpy().tolist()))
        return dict(rows=rows, metrics=check_metrics(rows))

    before_train = evaluate('train')
    atomic_json(out / 'before-train.json', before_train)
    started = time.monotonic()
    swap = psutil.swap_memory().used
    history = []
    checkpoints = []
    reason = 'update_cap'
    after_train = before_train
    for step in range(1, MAX_UPDATES + 1):
        if (deadline - datetime.now(timezone.utc)).total_seconds() <= FINAL_EVALUATION_RESERVE_S:
            reason = 'deadline_reserving_evaluation'
            break
        person = (step - 1) % 3
        batch = concatenate_observations([inputs.batch(datasets['train'][person * BATCH_SIZE + i]) for i in range(BATCH_SIZE)])
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        base = policy.get_base_model()
        actions = base.prepare_action(batch)
        training_noise, training_time = sample_training_conditions(base.model, actions.shape, actions.device, step, sampling)
        loss = sequence_loss(policy, batch, first_weight=1., gap_weight=1., noise=training_noise, time=training_time)
        if not torch.isfinite(loss):
            raise ValueError('学習損失が非有限です')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(list(named.values()), GRADIENT_LIMIT)
        if not torch.isfinite(gradient):
            raise ValueError('学習勾配が非有限です')
        optimizer.step()
        scheduler.step()
        torch.mps.synchronize()
        rss = psutil.Process().memory_info().rss
        driver = torch.mps.driver_allocated_memory()
        if rss + driver > MEMORY_LIMIT_BYTES or psutil.swap_memory().used - swap > SWAP_LIMIT_BYTES:
            raise RuntimeError('メモリまたはswapの上限に達しました')
        history.append(dict(step=step, person=person, loss=float(loss.detach().cpu()), gradient_norm=float(gradient.detach().cpu()), rss_bytes=rss, driver_bytes=driver,
                            time_mean=float(training_time.mean().detach().cpu()), endpoint_override=sampling == ENDPOINT_SAMPLING and step % ENDPOINT_INTERVAL == 0))
        if step % 10 == 0:
            atomic_json(out / 'progress.json', dict(completed=False, updates=step, last=history[-1]))
            print(json.dumps(history[-1]), flush=True)
        if step in CHECK_STEPS:
            after_train = evaluate('train')
            checkpoints.append(dict(step=step, metrics=after_train['metrics']))
            atomic_json(out / f'train-{step}.json', after_train)
            if after_train['metrics']['passed']:
                reason = 'first_train_pass'
                break
    updates = len(history)
    after_train = evaluate('train')
    final = out / 'adapter-final'
    policy.save_pretrained(final, safe_serialization=True)
    torch.save(dict(step=updates, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state(), history=history), final / 'training-state.pt')
    final_trainable = trainable_digest(named)
    after_validation = evaluate('validation')
    atomic_json(out / 'after-train.json', after_train)
    atomic_json(out / 'after-validation.json', after_validation)
    expected = after_train['rows'][0]['prediction']
    del optimizer, scheduler, named, policy
    gc.collect()
    torch.mps.empty_cache()
    policy = load_policy(adapter=initial, chunk_size=1)
    before_validation = evaluate('validation')
    atomic_json(out / 'before-validation.json', before_validation)
    del policy
    gc.collect()
    torch.mps.empty_cache()
    policy = load_policy(adapter=final, chunk_size=1)
    with torch.inference_mode():
        reloaded = inputs.denormalize_action(policy.predict_action_chunk(inputs.batch(model_observation(datasets['train'][0])), noise=noise)[0, 0])
    difference = float(np.max(np.abs(reloaded - expected)))
    atomic_json(out / 'result.json', dict(completed=True, updates=updates, stop_reason=reason,
        time_sampling=sampling, endpoint_updates=sum(row['endpoint_override'] for row in history),
        train_passed=after_train['metrics']['passed'], validation_passed=after_validation['metrics']['passed'],
        before_train=before_train['metrics'], after_train=after_train['metrics'],
        before_validation=before_validation['metrics'], after_validation=after_validation['metrics'],
        reload_max_abs_difference=difference, weights_changed=initial_trainable != final_trainable,
        adapter_sha256=digest(final / 'adapter_model.safetensors'), history=history, train_checks=checkpoints,
        seconds=time.monotonic() - started, adopted=False, full_task_success=False, evaluation_people_used=False))
    atomic_json(out / 'progress.json', dict(completed=True, updates=updates, result='result.json'))


if __name__ == '__main__':
    main()
