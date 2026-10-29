"""同期実演を公式ローダーで照合し、固定の小規模比較へ接続する。"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from export_panda_dataset import statistics
from paired_switch_data import training_pairs, MANIFEST_VERSION as PAIR_VERSION
from relative_joint_action import file_hash, build_normalization, apply_relative_train_scale, bind_policy
from relative_sequence_data import sequence_rows, mixed_batches, command_metrics
from relative_sequence_training import training_sample, evaluate, SPLITS, UPDATES, SEEDS
from resumable_job import atomic_json
from smolvla_runtime import load_policy, ModelInput
from eyedrop.smolvla_observation import command_observation_contract

SEED = 714
FLOAT32_TOLERANCE = 1e-6
EXPECTED_EPISODES = 6
OLD = Path('outputs/diagnostics/smolvla-command-timing-connection-02')
INITIAL_SHA = 'e0fb493a00f937e792b3506052901ae6fe07a73bcf2380933110a0f41fc08d67'


def completed_baselines(previous_root, out, protocol, labels):
    """教材・重みが同じ場合だけ、完了済みの学習前予測を引き継ぐ。"""
    reusable = {}
    previous=json.loads((previous_root/'protocol.json').read_bytes())
    for key in ('initial_model_sha256','initial_rng_sha256','raw','dataset','micro_source'):
        if previous[key]!=protocol[key]:
            raise ValueError('再利用する比較の条件が異なります: '+key)
    for name in ('samples.pt','labels.json','batches.json','relative-normalization.json'):
        if file_hash(previous_root/name)!=file_hash(out/name):
            raise ValueError('再利用する比較の教材が異なります: '+name)
    for seed in SEEDS:
        for split in SPLITS:
            source=previous_root/f'before-{seed}-{split}.json'
            if source.exists():
                value=json.loads(source.read_bytes())
                for group in ('sequence','micro'):
                    if len(value[group]['rows'])!=len(labels[split][group]):
                        raise ValueError('再利用する予測の件数が異なります')
                reusable[seed,split]=source
    atomic_json(out/'reuse.json',dict(source=str(previous_root),
        completed_predictions={str(p):file_hash(p) for p in reusable.values()},
        checked=['initial weights','rng','raw','dataset','micro source','samples','labels','batches','normalization'],
        reason='reuse complete before-training predictions after bounded-stage timeout; no weight updates'))
    return reusable


def main():
    parser = argparse.ArgumentParser()
    for key in ('raw', 'dataset', 'budget', 'output'):
        parser.add_argument(key)
    parser.add_argument('--reuse-baseline', type=Path)
    args = parser.parse_args()
    raw, data, out = map(Path, (args.raw, args.dataset, args.output))
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads((raw/'protocol.json').read_bytes())
    collected = json.loads((raw/'result.json').read_bytes())
    qa = json.loads((data/'result.json').read_bytes())
    if not collected['accepted'] or not qa['accepted'] or qa['episodes'] != dict.fromkeys(SPLITS, EXPECTED_EPISODES):
        raise ValueError('固定した学習6・検査6の教材品質が必要です')
    for person in ('original', 'seed-san', 'godette'):
        audit=json.loads((raw/'audit'/person/'result.json').read_bytes())
        if not audit['completed'] or any(v['accepted']!=4 or v['count']!=4 for v in audit['trials'].values()):
            raise ValueError('記録した教師の厳格な独立監査が不合格です')
    old_quality = json.loads((OLD/'result.json').read_bytes())
    old_protocol = json.loads((OLD/'protocol.json').read_bytes())
    old_norm=json.loads((OLD/'relative-normalization.json').read_bytes())
    if old_norm.get('observation_history_contract')!=command_observation_contract():
        raise ValueError('既存の開閉対照と同期教材の自己状態の意味が一致しません')
    hashes = {}

    def remember(path):
        value = file_hash(path); hashes[str(path)] = value
        return value

    for name, key in (('samples.pt','samples_sha256'), ('labels.json','labels_sha256')):
        if remember(OLD/name) != old_quality[key]:
            raise ValueError('既存の開閉対照が保存時から変化しています')
    initial = Path(old_protocol['initial'])
    if remember(initial/'policy/model.safetensors') != INITIAL_SHA:
        raise ValueError('固定初期重みが一致しません')
    if remember(Path(old_protocol['rng'])/'initial-rng.pt') != old_protocol['initial_rng_sha256']:
        raise ValueError('固定乱数状態が一致しません')
    old_cache = torch.load(OLD/'samples.pt', map_location='cpu', weights_only=True)
    old_cache = {split:{'micro':old_cache[split]['micro']} for split in SPLITS}
    old_labels = json.loads((OLD/'labels.json').read_bytes())
    cache, labels, rows_by_split = {}, {}, {}
    train_states, train_actions, train_relative = [], [], []
    selected_images = {split:set() for split in SPLITS}
    tensor_images = {split:set() for split in SPLITS}
    for split in SPLITS:
        dataset = LeRobotDataset(f'local/eyedrop-panda-{split}', root=data/split,
                                 delta_timestamps={'action':[0.]}, video_backend='pyav')
        cache[split] = {'sequence':{}, 'micro':old_cache[split]['micro']}
        labels[split] = {'sequence':[], 'micro':old_labels[split]['micro']}
        if len(labels[split]['micro']) != 9:
            raise ValueError('既存の開閉9対照が不足しています')
        for episode, item in enumerate(plan[split]):
            folder = raw/item['id']
            record = json.loads((folder/'result.json').read_bytes())
            source = Path(record['original_source'])
            if remember(source/'trace.json') != record['source_trace_sha256']:
                raise ValueError('元の次指令が変化しました')
            remember(folder/'frames.json')
            frames = json.loads((folder/'frames.json').read_bytes())
            trace = json.loads((source/'trace.json').read_bytes())
            selected = sequence_rows(frames, trace, split=split, episode=episode, case_id=item['id'])
            for row in selected:
                index = row['dataset_index']
                sample = training_sample(dataset[index])
                frame = frames[row['step']]
                np.testing.assert_allclose(sample['observation.state'].numpy(), frame['state'], atol=FLOAT32_TOLERANCE, rtol=0)
                np.testing.assert_allclose(sample['action'][0].numpy(), frame['action'], atol=FLOAT32_TOLERANCE, rtol=0)
                for camera, side in enumerate(('left','right'), 1):
                    image_path = raw/frame[side]; selected_images[split].add(remember(image_path))
                    with Image.open(image_path) as image:
                        pixels = np.asarray(image.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
                    np.testing.assert_array_equal(sample[f'observation.images.camera{camera}'].numpy(), pixels)
                    tensor_images[split].add(hashlib.sha256(pixels.tobytes()).hexdigest())
                cache[split]['sequence'][index] = sample
            labels[split]['sequence'].extend(selected)
            if split == 'train':
                train_states.extend(r['state'] for r in frames)
                train_actions.extend(r['action'] for r in frames)
                train_relative.extend((np.array(r['action'][:7])-r['state'][:7]).tolist() for r in frames)
        for sample in cache[split]['micro'].values():
            for camera in (1,2):
                tensor_images[split].add(hashlib.sha256(sample[f'observation.images.camera{camera}'].numpy().tobytes()).hexdigest())
            if split == 'train':
                train_states.append(sample['observation.state'].tolist())
                train_actions.append(sample['action'][0].tolist())
                train_relative.append((sample['action'][0,:7]-sample['observation.state'][:7]).tolist())
        rows_by_split[split] = labels[split]['sequence']
        metrics = command_metrics([dict(row,prediction=row['reference']) for row in rows_by_split[split]])
        atomic_json(out/f'teacher-{split}.json', metrics)
        if not metrics['passed']:
            raise ValueError('教師自身が指令採点を通りません')
    if selected_images['train'] & selected_images['validation'] or tensor_images['train'] & tensor_images['validation']:
        raise ValueError('分割間の画像重複があります')
    base = dict(source_split='train',language_instruction=plan['language_instruction'],
                observation_history_contract=command_observation_contract(),
                features={'observation.state':statistics(train_states), 'action':statistics(train_actions)})
    atomic_json(out/'absolute-normalization.json', base)
    norm = build_normalization(base, file_hash(out/'absolute-normalization.json'))
    train_hashes = {str(p):remember(p) for p in (data/'train/data').rglob('*.parquet')}
    scale = dict(completed=True,updates=0,scale_source_train_only=True,input_hashes=train_hashes,
                 current_joint_scale=norm['features']['action']['std'][:7],
                 relative_train_std=statistics(train_relative)['std'],
                 auxiliary_micro_sha256=old_quality['samples_sha256'])
    atomic_json(out/'scale-audit.json', scale)
    norm = apply_relative_train_scale(norm, scale, file_hash(out/'scale-audit.json'))
    atomic_json(out/'relative-normalization.json', norm)
    atomic_json(out/'rows.json', rows_by_split)
    atomic_json(out/'batches.json', mixed_batches(rows_by_split['train'], updates=UPDATES, batch_size=3, seed=SEED))
    torch.save(cache,out/'samples.pt'); atomic_json(out/'labels.json',labels)
    protocol = dict(initial=str(initial), initial_model_sha256=INITIAL_SHA, rng=old_protocol['rng'],
                    initial_rng_sha256=old_protocol['initial_rng_sha256'], manifest=str(out), dataset=str(data),
                    micro=old_protocol['micro'], raw=str(raw), parent_budget=args.budget,
                    hypothesis='downward synchronized demonstrations and reviewed flow-grid objective',
                    micro_source=str(OLD), micro_is_historical_retention=True, validation_used_for_selection=False,
                    initial_origin='official SmolVLA-derived existing merged 600-update adapter; not untouched public checkpoint',
                    evaluation_scope='held out from this training cycle; previously used development conditions, not first-use final test',
                    source_sha256=file_hash(__file__),
                    environment={name:importlib.metadata.version(name) for name in ('lerobot','torch','numpy','pillow')})
    atomic_json(out/'protocol.json',protocol)
    pair_manifest = dict(schema=PAIR_VERSION,source_split='train',labels_sha256=file_hash(out/'labels.json'),
                         pairs=training_pairs(labels['train']['sequence']))
    atomic_json(out/'pairs.json',pair_manifest)
    reusable = completed_baselines(args.reuse_baseline,out,protocol,labels) if args.reuse_baseline else {}
    policy = load_policy(full_policy=initial/'policy',native_full=True,chunk_size=1)
    bind_policy(policy,norm);inputs=ModelInput(policy,out/'relative-normalization.json')
    maximum = 0.
    for split in SPLITS:
        for source in cache[split]:
            for sample in cache[split][source].values():
                encoded=inputs.batch(sample)['action'][0,0]
                decoded=inputs.denormalize_action(encoded,reference_state=sample['observation.state'])
                maximum=max(maximum,float(abs(decoded-sample['action'][0].numpy()).max()))
    if maximum > FLOAT32_TOLERANCE:
        raise ValueError('教師の往復変換が一致しません')
    for seed in SEEDS:
        noise=torch.randn(1,1,32,generator=torch.Generator().manual_seed(seed)).to('mps')
        for split in SPLITS:
            if (seed,split) in reusable:
                source=reusable[seed,split]
                result=json.loads(source.read_bytes())
                shutil.copyfile(source,out/f'before-{seed}-{split}.json')
            else:
                result=evaluate(policy,inputs,cache,labels,split,noise)
                atomic_json(out/f'before-{seed}-{split}.json',result)
            if seed==SEED:atomic_json(out/f'before-relative-{split}.json',result)
            print(json.dumps(dict(stage='before',seed=seed,split=split,sequence=result['sequence']['metrics']['passed'])),flush=True)
    quality=dict(completed=True,ready_for_training=True,updates=0,teacher_roundtrip_max=maximum,
                 samples_sha256=file_hash(out/'samples.pt'),labels_sha256=file_hash(out/'labels.json'),
                 batches_sha256=file_hash(out/'batches.json'),helper_sha256=file_hash(Path(__file__).with_name('relative_sequence_training.py')),
                 normalization_sha256={'relative':file_hash(out/'relative-normalization.json')},
                 files_sha256=hashes,seconds=time.monotonic()-started,sequence_counts={s:len(rows_by_split[s]) for s in SPLITS},
                 scope='data integrity and base model connection; flow-grid gradient connection still required')
    atomic_json(out/'result.json',quality)
    fixed={key:quality[key] for key in ('samples_sha256','labels_sha256','batches_sha256')}
    fixed.update(connection=str(out),normalization_sha256=quality['normalization_sha256']['relative'],
                 initial_model_sha256=INITIAL_SHA,initial_rng_sha256=protocol['initial_rng_sha256'])
    atomic_json(out/'pair-proposal.json',dict(paired_switch_objective='paired-flow-grid',fixed=fixed))


if __name__ == '__main__':
    main()
