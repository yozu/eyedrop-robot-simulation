"""画像2枚と自己状態32成分だけを受け取るMac推論ワーカー。"""
import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from smolvla_runtime import load_policy,ModelInput
from direct_gripper import observation_features
from tree_gripper import TreeGripper
from cartesian_action import action_dimension,JOINT_DIM
from endpoint_policy import validate_endpoint_policy,MODE as ENDPOINT_MODE
from relative_joint_action import MODE as RELATIVE,verify_initial_policy,bind_policy

ACTION_STEPS = (1, 2, 5)
ACTION_CHUNK = 50
PADDED_ACTION_DIM = 32
NOISE_MODES = ('fresh', 'fixed', 'zero')


def step_noise(generator, mode, previous, chunk_size=ACTION_CHUNK):
    if mode not in NOISE_MODES:
        raise ValueError('推論noiseの方式が比較条件外です')
    if mode == 'zero':
        return torch.zeros(1, chunk_size, PADDED_ACTION_DIM)
    if mode == 'fixed' and previous is not None:
        return previous
    return torch.randn(1, chunk_size, PADDED_ACTION_DIM, generator=generator)


def select_prediction(policy, batch, noise, action_steps):
    if action_steps not in ACTION_STEPS:
        raise ValueError('行動列の長さが比較条件外です')
    if action_steps == 1:
        return policy.predict_action_chunk(batch, noise=noise)[0, 0]
    return policy.select_action(batch, noise=noise)[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('normalization')
    checkpoint=parser.add_mutually_exclusive_group();checkpoint.add_argument('--adapter');checkpoint.add_argument('--full-policy')
    parser.add_argument('--action-steps',type=int,choices=ACTION_STEPS,default=1)
    parser.add_argument('--noise-mode',choices=NOISE_MODES,default='fresh')
    parser.add_argument('--gripper-tree')
    parser.add_argument('--prediction-chunk-size',type=int,choices=(1,50))
    parser.add_argument('--endpoint-regression',action='store_true')
    parser.add_argument('--relative-initial-sha256')
    args=parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        normalization=Path(args.normalization)
        document=json.loads(normalization.read_bytes());action_dim=action_dimension(document)
        if args.relative_initial_sha256:
            if not args.full_policy or args.prediction_chunk_size!=1 or args.action_steps!=1 or args.gripper_tree or args.endpoint_regression:
                raise ValueError('相対指令の初期比較は通常重み・1予測1実行の既存flowだけです')
            verify_initial_policy(args.full_policy,document,args.relative_initial_sha256)
        policy=load_policy(adapter=args.adapter,full_policy=args.full_policy,action_dim=action_dim,chunk_size=args.prediction_chunk_size)
        if args.relative_initial_sha256:bind_policy(policy,document)
        inputs=ModelInput(policy,normalization)
        if inputs.joint_encoding==RELATIVE and (args.action_steps!=1 or args.gripper_tree or args.endpoint_regression):
            raise ValueError('相対関節は1予測1実行・既存flowのみで、別指幅判定器を併用しません')
        if args.endpoint_regression != (args.noise_mode=='zero') or (args.endpoint_regression and args.action_steps!=1):
            raise ValueError('1段回帰はzero noise・1動作だけで実行してください')
        validate_endpoint_policy(policy,args.full_policy,requested=args.endpoint_regression)
        tree=None
        if args.gripper_tree:
            if action_dim!=JOINT_DIM:raise ValueError('既存指幅木は関節8次元モデル専用です')
            if not args.full_policy: raise ValueError('画像expert分類木には対応する通常モデルが必要です')
            tree=TreeGripper(json.loads(Path(args.gripper_tree).read_text()))
            tree.verify_bindings(Path(args.normalization),Path(args.full_policy)/'model.safetensors')
    if args.action_steps>policy.config.chunk_size:raise ValueError('実行する動作数が学習した予測数を超えています')
    policy.config.n_action_steps=args.action_steps
    generator=torch.Generator().manual_seed(714)
    previous_noise=None
    print(json.dumps({'ready':True,'adapter':args.adapter,'full_policy':args.full_policy,'relative_initial_sha256':args.relative_initial_sha256,'action_steps':args.action_steps,'noise_mode':args.noise_mode,'gripper_tree':args.gripper_tree,'robot_action_space':inputs.robot_action_space,'action_dim':action_dim,'prediction_chunk_size':policy.config.chunk_size,'policy_mode':ENDPOINT_MODE if args.endpoint_regression else 'smolvla-flow-matching-v1','num_steps':policy.config.num_steps}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request['operation']=='close':break
            if request['operation']=='reset':
                generator.manual_seed(request['seed']);previous_noise=None;policy.reset();print('{}',flush=True);continue
            if set(request)!={'operation','state','left','right'}:raise ValueError('推論の入力列が契約外です')
            if len(request['state'])!=32 or not np.isfinite(request['state']).all():raise ValueError('自己状態が不正です')
            sample={'observation.state':request['state']}
            for i,side in enumerate(('left','right'),1):
                image=Image.open(io.BytesIO(base64.b64decode(request[side]))).convert('RGB')
                sample[f'observation.images.camera{i}']=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/255
            batch=inputs.batch(sample)
            # 比較時の乱数列は物理時点ごとに進め、補充時だけ公式キューが使用する。
            previous_noise=step_noise(generator,args.noise_mode,previous_noise,chunk_size=policy.config.chunk_size)
            noise=previous_noise.to('mps')
            with torch.no_grad():prediction=select_prediction(policy,batch,noise,args.action_steps)
            action=inputs.denormalize_action(prediction,reference_state=request['state']).tolist()
            if tree is not None:
                feature=observation_features(policy,batch)[0].numpy()
                action[7]=tree.predict(feature)
            if len(action)!=action_dim or not np.isfinite(action).all():raise ValueError('推論の次元または値が不正です')
            print(json.dumps({'action':action}),flush=True)
        except Exception as error:
            print(json.dumps({'error':repr(error)}),flush=True)


if __name__=='__main__':main()
