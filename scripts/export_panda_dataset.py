"""教師品質・履歴・分割を監査してLeRobotDatasetへ保存する。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from eyedrop.smolvla_observation import PressHistory, history_from_normalization
from resumable_job import atomic_json

STATE_DIM=32
ACTION_DIM=8
FPS=20
IMAGE_SIZE=512
INSTRUCTION="Place one drop in the right eye, release and retreat."
CONSTANT_STD_EPSILON=1e-8
SUPPORTED_SPLIT_COUNTS=((6,2),(12,2),(6,6))


def validate_episode_splits(protocol,episodes):
    counts={split:len(protocol[split]) for split in ('train','validation')}
    expected=[(row['id'],split) for split in counts for row in protocol[split]]
    actual=[(row['id'],row['split']) for row in episodes]
    if (tuple(counts.values()) not in SUPPORTED_SPLIT_COUNTS
        or len({name for name,_ in expected})!=len(expected) or actual!=expected):
        raise ValueError('事前定義した実演数・分割・順序と異なります')
    return counts


def statistics(values):
    values=np.asarray(values,dtype=np.float64)
    deviation=values.std(axis=0)
    return {'mean':values.mean(axis=0).tolist(),
            'std':np.where(deviation<CONSTANT_STD_EPSILON,1.,deviation).tolist(),
            'constant_dimensions':np.where(deviation<CONSTANT_STD_EPSILON)[0].tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw');parser.add_argument('output')
    parser.add_argument('--instruction-file')
    args=parser.parse_args();raw=Path(args.raw).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    collected=json.loads((raw/'result.json').read_text())
    protocol=json.loads((raw/'protocol.json').read_text())
    instruction=protocol['language_instruction'] if 'language_instruction' in protocol else INSTRUCTION
    if args.instruction_file:instruction=Path(args.instruction_file).read_text(encoding='utf8').strip()
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('共通文章指示が不正です')
    rest_gap=protocol.get('rest_gap_m',.0199)
    if not collected['accepted']:raise RuntimeError('全実演の品質合格が必要です')
    validate_episode_splits(protocol,collected['episodes'])
    records={'train':[],'validation':[]};hashes={'train':set(),'validation':set()}
    features={
        'observation.state':{'dtype':'float32','shape':(STATE_DIM,),'names':[f'own_{i}' for i in range(STATE_DIM)]},
        'action':{'dtype':'float32','shape':(ACTION_DIM,),'names':[f'command_{i}' for i in range(ACTION_DIM)]},
        **{f'observation.images.camera{i}':{'dtype':'image','shape':(IMAGE_SIZE,IMAGE_SIZE,3),'names':['height','width','channels']} for i in (1,2)}}
    count=0
    for episode in collected['episodes']:
        split=episode['split'];rows=json.loads((raw/episode['id']/'frames.json').read_text())
        initial_command=episode.get('initial_gap_command_m',rest_gap)
        history=history_from_normalization(protocol,rest_gap);history.reset(rows[0]['state'][:16],initial_command)
        for index,row in enumerate(rows):
            if set(row)!={'step','time','state','action','left','right'}:raise ValueError('データの列契約が変化しました')
            if index:history.append(index,row['state'][:16],rows[index-1]['action'][7])
            if row['step']!=index or abs(row['time']-index/FPS)>1e-9:raise ValueError('時刻の不整合')
            np.testing.assert_allclose(row['state'],history.observation(),atol=1e-9,rtol=0)
            if len(row['action'])!=ACTION_DIM or not np.isfinite(row['action']).all():raise ValueError('動作が不正です')
            for side in ('left','right'):
                file=raw/row[side]
                with Image.open(file) as image:
                    if image.size!=(IMAGE_SIZE,IMAGE_SIZE) or image.mode not in ('RGB','RGBA'):raise ValueError('画像形式が不正です')
                hashes[split].add(hashlib.sha256(file.read_bytes()).hexdigest())
            count+=1
        records[split].append(rows)
    overlap=hashes['train']&hashes['validation']
    if overlap:raise ValueError(f'分割間の画像重複: {len(overlap)}')
    normalizer={key:statistics([row[column] for episode in records['train'] for row in episode])
                for key,column in [('observation.state','state'),('action','action')]}
    atomic_json(output/'normalization.json',{'source_split':'train','features':normalizer,'language_instruction':instruction,
        **({'observation_history_contract':protocol['observation_history_contract']} if 'observation_history_contract' in protocol else {})})
    for split in ('train','validation'):
        dataset=LeRobotDataset.create(repo_id=f'local/eyedrop-panda-{split}',fps=FPS,features=features,
                                      root=output/split,robot_type='panda-eyedrop-sim',use_videos=False,
                                      image_writer_processes=0,image_writer_threads=1,video_backend='pyav')
        try:
            for rows in records[split]:
                for row in rows:
                    frame={'observation.state':np.asarray(row['state'],dtype=np.float32),
                           'action':np.asarray(row['action'],dtype=np.float32),'task':instruction}
                    for i,side in enumerate(('left','right'),1):
                        with Image.open(raw/row[side]) as image:frame[f'observation.images.camera{i}']=np.asarray(image.convert('RGB'))
                    dataset.add_frame(frame)
                dataset.save_episode()
        finally:dataset.finalize()
    # 公式ローダーで50操作の未来ラベルと末尾paddingを確認する。
    loaded=LeRobotDataset('local/eyedrop-panda-train',root=output/'train',delta_timestamps={'action':[i/FPS for i in range(50)]},video_backend='pyav')
    sample=loaded[0]
    shapes={key:list(sample[key].shape) for key in features}
    if shapes['action']!=[50,ACTION_DIM] or shapes['observation.state']!=[STATE_DIM]:raise ValueError('公式ローダーの出力契約不一致')
    result={'accepted':True,'episodes':{k:len(v) for k,v in records.items()},'frames':count,
            'episode_frames':protocol['frames'],'fps':FPS,'language_instruction':instruction,
            'instruction_source_sha256':hashlib.sha256(Path(args.instruction_file).read_bytes()).hexdigest() if args.instruction_file else None,
            'cross_split_image_duplicates':len(overlap),'history_replay':'all frames match',
            'normalization_source':'train only','sample_shapes':shapes,'seconds':time.monotonic()-started}
    atomic_json(output/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
