"""固定した連続実演と開閉対照の接続。教師情報は採点だけに使う。"""
from copy import deepcopy

UPDATES = 3000
TRAIN_KEYS = ('observation.state', 'observation.images.camera1', 'observation.images.camera2',
              'action', 'action_is_pad')
SOURCES = ('micro', 'sequence')
SPLITS = ('train', 'validation')
SEEDS = (714, 715, 716, 717, 718)


def training_sample(sample):
    return {key: sample[key] for key in TRAIN_KEYS}


def resolve_batch(train_cache, descriptors):
    if any(row['source'] not in SOURCES for row in descriptors):
        raise ValueError('未登録の教材です')
    return [training_sample(train_cache[row['source']][row['index']]) for row in descriptors]


def fixed_training_complete(updates):
    if type(updates) is not int or not 0 <= updates <= UPDATES:
        raise ValueError('更新数が契約範囲外です')
    return updates == UPDATES


def compatible_norms(left, right):
    docs = [deepcopy(value) for value in (left, right)]
    for value in docs:
        value['features']['action']['std'][:7] = [None] * 7
        value['relative_action_contract'].pop('scale_source', None)
        value['relative_action_contract'].pop('scale_audit', None)
    return docs[0] == docs[1]


def evaluate(policy, inputs, cache, labels, split, noise):
    import torch
    from relative_sequence_data import command_metrics
    from train_smolvla_micro_conditioning import check_metrics, model_observation, OPEN_MINIMUM, PRESS_GAP_M
    policy.eval()
    results = {}
    with torch.inference_mode():
        for source in SOURCES:
            rows = []
            for label in labels[split][source]:
                index = label['index'] if source == 'micro' else label['dataset_index']
                sample = cache[split][source][index]
                prediction = inputs.denormalize_action(policy.predict_action_chunk(inputs.batch(model_observation(sample)),
                    noise=noise)[0, 0], reference_state=sample['observation.state'])
                row = dict(label, prediction=prediction.tolist(), reference=sample['action'][0].numpy().tolist())
                rows.append(row)
            metrics = check_metrics(rows) if source == 'micro' else command_metrics(rows)
            if source == 'micro':
                metrics['false_press'] = sum(r['actual_open'] < OPEN_MINIMUM and r['prediction'][7] < PRESS_GAP_M for r in rows)
                metrics['passed'] &= metrics['false_press'] == 0
            results[source] = dict(rows=rows, metrics=metrics)
    return results
