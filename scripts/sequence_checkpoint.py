"""固定連続教材の途中保存を、全ファイルの確定後に公開する。"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from lerobot.utils.random_utils import serialize_rng_state,deserialize_rng_state

from relative_joint_action import file_hash,save_policy_binding
from relative_sequence_training import UPDATES
from resumable_job import atomic_json

CHECKPOINT_INTERVAL = 1000
SAVE_RESERVE_SECONDS = 180
DISK_RESERVE_BYTES = 512 * 1024**2
FORMAT_VERSION = 1
REQUIRED_FILES = ('model.safetensors','config.json','eyedrop-action-contract.json',
                  'normalization.json','training-state.pt','history.json','protocol.json')


def _tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(_tensor_bytes(v) for v in value)
    return 0


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def save_checkpoint(output,policy,optimizer,scheduler,normalization,history,protocol,pair_rng=None):
    steps=len(history)
    if not 0<steps<=UPDATES or [r['step'] for r in history]!=list(range(1,steps+1)):
        raise ValueError('途中保存の更新履歴が不正です')
    root=Path(output)/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    destination=root/f'step-{steps:06d}'
    if destination.exists():raise FileExistsError(destination)
    required=_tensor_bytes(policy.state_dict())+_tensor_bytes(optimizer.state_dict())+DISK_RESERVE_BYTES
    if shutil.disk_usage(root).free<required:raise OSError('途中保存のための空き容量が不足しています')
    rng=serialize_rng_state()
    state=dict(updates=steps,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               cpu_rng=rng['torch_rng_state'],rng_state=rng)
    if 'torch_mps_rng_state' in rng:state['mps_rng']=rng['torch_mps_rng_state']
    if pair_rng is not None:state['pair_rng']=pair_rng.get_state()
    temporary=Path(tempfile.mkdtemp(prefix=f'.{destination.name}.pending-',dir=root))
    try:
        policy.save_pretrained(temporary);save_policy_binding(policy,temporary)
        (temporary/'normalization.json').write_bytes(Path(normalization).read_bytes())
        torch.save(state,temporary/'training-state.pt')
        atomic_json(temporary/'history.json',history);atomic_json(temporary/'protocol.json',protocol)
        hashes={name:file_hash(temporary/name) for name in REQUIRED_FILES}
        atomic_json(temporary/'checkpoint.json',dict(version=FORMAT_VERSION,updates=steps,files=hashes))
        for path in temporary.iterdir():
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        _sync_directory(temporary)
        # 同一ファイルシステム内で、完了した保存点だけを公開する。
        os.rename(temporary,destination);_sync_directory(root)
    finally:
        deserialize_rng_state(rng)
        if pair_rng is not None:pair_rng.set_state(state['pair_rng'])
    return destination


def inspect_checkpoint(directory):
    directory=Path(directory)
    if directory.name.startswith('.') or not (directory/'checkpoint.json').is_file():
        raise ValueError('未完了の途中保存は読み込めません')
    record=json.loads((directory/'checkpoint.json').read_bytes())
    if record.get('version')!=FORMAT_VERSION or set(record.get('files',{}))!=set(REQUIRED_FILES):
        raise ValueError('途中保存の形式が一致しません')
    for name,expected in record['files'].items():
        if not (directory/name).is_file() or file_hash(directory/name)!=expected:
            raise ValueError('途中保存の完全性が不一致です: '+name)
    protocol=json.loads((directory/'protocol.json').read_bytes())
    if record['files']['normalization.json']!=protocol['normalization_sha256']:
        raise ValueError('途中保存と学習の正規化が異なります')
    history=json.loads((directory/'history.json').read_bytes());steps=record['updates']
    if type(steps) is not int or not 0<steps<=UPDATES or [r['step'] for r in history]!=list(range(1,steps+1)):
        raise ValueError('途中保存の更新数・履歴が不一致です')
    return dict(history=history,protocol=protocol,record=record)


def load_checkpoint(directory):
    directory=Path(directory);saved=inspect_checkpoint(directory)
    # Tensorを復元する前に、全構成ファイルのハッシュを照合する。
    state=torch.load(directory/'training-state.pt',map_location='cpu',weights_only=True)
    steps=saved['record']['updates']
    if state['updates']!=steps or state['scheduler']['last_epoch']!=steps:
        raise ValueError('更新数・履歴・schedulerが対応していません')
    return dict(state=state,**saved)


def restore_training_state(state,optimizer,scheduler,pair_rng=None):
    if (pair_rng is not None)!=('pair_rng' in state):
        raise ValueError('対の学習用乱数の有無が不一致です')
    scheduler.load_state_dict(state['scheduler']);optimizer.load_state_dict(state['optimizer'])
    deserialize_rng_state(state['rng_state'])
    if pair_rng is not None:pair_rng.set_state(state['pair_rng'])


def resume_contract(saved,previous,parent,now,*,source_sha256,scale):
    protocol=saved['protocol']
    matches=[i for i,row in enumerate(parent.get('candidates',[])) if Path(row['output']).resolve()==Path(previous).resolve()]
    if (not parent.get('user_authorized') or len(matches)!=1 or
        protocol['source_sha256']!=source_sha256 or protocol['scale']!=scale or
        parent.get('candidate_max_updates')!=UPDATES):
        raise ValueError('同じ候補・ソース・尺度・許可枠でのみ再開できます')
    index=matches[0];candidate=parent['candidates'][index]
    if (candidate.get('checkpoint_resumes_started',0)!=0 or candidate['scale']!=scale or
        not 0<parent['new_training_candidates_started']<=parent['new_training_candidates_cap']):
        raise ValueError('同じ候補の途中保存再開は1回だけです')
    deadline=min(datetime.fromisoformat(protocol['deadline']),datetime.fromisoformat(parent['compute_until']))
    if (deadline-now).total_seconds()<=SAVE_RESERVE_SECONDS:
        raise ValueError('元の期限内に保存と再開を完了できません')
    return dict(candidate_index=index,deadline=deadline.isoformat(),offset=saved['record']['updates'])
