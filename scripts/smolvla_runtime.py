"""G0aで確認した事前学習重みと実データの正規化を接続する。"""
import json
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import snapshot_download
from lerobot.configs import FeatureType,PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
from peft import PeftModel,LoraConfig
from safetensors.torch import load_model as load_safetensors_model
from check_smolvla_g0a import MODEL_ID,MODEL_REVISION,VLM_ID,VLM_REVISION,STATE_DIM,ACTION_DIM,IMAGE_SIZE,LORA_RANK,SEED
from action_encoding import encode,decode,LINEAR,BOUNDED
from joint_action_basis import ABSOLUTE,ORTHOGONAL,checked_basis,encode_joints,decode_joints
from cartesian_action import action_dimension, encode_cartesian, decode_cartesian, CARTESIAN_SPACE, JOINT_SPACE, CARTESIAN_DIM, JOINT_DIM
from relative_joint_action import (MODE as RELATIVE, relative_action, absolute_action,
                                   validate_policy_binding, read_policy_binding)
from eyedrop.smolvla_observation import command_observation_contract

INSTRUCTION='Place one drop in the right eye, release and retreat.\n'


def native_trainable_name(name):
    return name.startswith(('model.vlm_with_expert.lm_expert.','model.state_proj.',
                            'model.action_in_proj.','model.action_out_proj.',
                            'model.action_time_mlp_in.','model.action_time_mlp_out.'))


def load_policy(adapter=None,train=False,full_projections=False,full_policy=None,native_full=False,chunk_size=None,action_dim=None):
    if adapter and (full_policy or native_full):raise ValueError('PEFTと通常モデルを同時には指定できません')
    if native_full and full_projections:raise ValueError('学習範囲は一つだけ指定してください')
    if full_policy and train and not native_full:raise ValueError('通常モデルの学習には明示した学習範囲が必要です')
    if not torch.backends.mps.is_available():raise RuntimeError('MPSが必要です')
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.mps.manual_seed(SEED)
    base=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
                                allow_patterns=['config.json','model.safetensors']))
    vlm=Path(snapshot_download(VLM_ID,revision=VLM_REVISION,local_files_only=True,
                               allow_patterns=['*.json','*.jinja','*.jinja2','*.txt','tokenizer.model']))
    source=Path(full_policy).resolve() if full_policy else base
    # 不一致の重みは大きなモデルを構築する前に拒否する。
    binding=read_policy_binding(source) if full_policy else None
    cfg=SmolVLAConfig.from_pretrained(source)
    if action_dim is not None and action_dim not in (CARTESIAN_DIM,JOINT_DIM):raise ValueError('未対応の動作次元です')
    if full_policy and action_dim is not None and cfg.action_feature.shape[0]!=action_dim:raise ValueError('保存重みの動作次元が異なります')
    dimension=cfg.action_feature.shape[0] if full_policy else (ACTION_DIM if action_dim is None else action_dim)
    if adapter and dimension!=JOINT_DIM:raise ValueError('既存PEFTは関節8次元専用です')
    if chunk_size is not None:
        if chunk_size not in (1,50):raise ValueError('予測する動作数は1または50です')
        if full_policy and cfg.chunk_size!=chunk_size:raise ValueError('保存重みの動作数と指定が異なります')
        cfg.chunk_size=chunk_size
    cfg.device='cpu';cfg.use_amp=False;cfg.push_to_hub=False;cfg.load_vlm_weights=False
    cfg.vlm_model_name=str(vlm);cfg.pretrained_path=source;cfg.n_action_steps=1
    if native_full:
        cfg.train_state_proj=True;cfg.train_expert_only=True;cfg.freeze_vision_encoder=True;cfg.use_peft=False
    cfg.input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(STATE_DIM,)),
                        **{f'observation.images.camera{i}':PolicyFeature(type=FeatureType.VISUAL,shape=(3,IMAGE_SIZE,IMAGE_SIZE)) for i in (1,2)}}
    cfg.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(dimension,))}
    if full_policy:
        # 構築直後のexpertがbfloat16でも、保存したfloat32を丸めて取り込まない。
        policy=SmolVLAPolicy(cfg).to(dtype=torch.float32)
        load_safetensors_model(policy,str(source/'model.safetensors'),strict=True,device='cpu')
    else:policy=SmolVLAPolicy.from_pretrained(source,config=cfg,strict=True,local_files_only=True)
    policy.config.load_vlm_weights=True;policy.config.device='mps'
    policy=policy.to(device='mps',dtype=torch.float32)
    if native_full:
        for name,parameter in policy.named_parameters():parameter.requires_grad_(train and native_trainable_name(name))
    elif full_policy:
        for parameter in policy.parameters():parameter.requires_grad_(False)
    elif adapter:policy=PeftModel.from_pretrained(policy,adapter,is_trainable=train).to('mps')
    elif train:
        if full_projections:
            config=LoraConfig(r=LORA_RANK,lora_alpha=LORA_RANK,bias='none',
                              target_modules=r'(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(action_time_mlp_in|action_time_mlp_out))',
                              modules_to_save=['model.state_proj','model.action_in_proj','model.action_out_proj'])
            policy=policy.wrap_with_peft(peft_config=config)
        else:policy=policy.wrap_with_peft(peft_cli_overrides={'method_type':'LORA','r':LORA_RANK,'lora_alpha':LORA_RANK})
    if full_policy:
        if read_policy_binding(source)!=binding:
            raise ValueError('構築中に相対指令の重み・設定が変更されました')
        if binding is not None:
            policy._eyedrop_action_contract={k:binding[k] for k in ('mode','normalization_sha256')}
    return policy.train() if train else policy.eval()


class ModelInput:
    def __init__(self,policy,normalization):
        document=json.loads(Path(normalization).read_text());self.stats=document['features']
        if ('observation_history_contract' in document and
            document['observation_history_contract'] != command_observation_contract()):
            raise ValueError('観測履歴の契約が未対応です')
        # 旧保存データは当時の固定文を維持し、新しいデータでは指示を明示する。
        self.instruction=document['language_instruction'] if 'language_instruction' in document else INSTRUCTION
        if not isinstance(self.instruction,str) or not self.instruction.strip():
            raise ValueError('共通の文章指示は空でない文字列として保存してください')
        self.action_dim=action_dimension(document);self.robot_action_space=document.get('robot_action_space',JOINT_SPACE)
        if policy.config.action_feature.shape[0]!=self.action_dim:raise ValueError('モデルと統計の動作次元が異なります')
        # 従来の正規化ファイルは線形表現。新しい表現は明示されたものだけを受け付ける。
        self.action_encoding=document['action_encoding'] if 'action_encoding' in document else LINEAR
        if self.action_encoding not in (LINEAR,BOUNDED):raise ValueError('未対応の動作表現です')
        self.joint_encoding=document.get('joint_action_encoding',ABSOLUTE)
        if self.joint_encoding not in (ABSOLUTE,ORTHOGONAL,RELATIVE):raise ValueError('未対応の関節座標です')
        validate_policy_binding(policy,document)
        self.joint_basis=document.get('joint_action_basis')
        if self.joint_encoding==ORTHOGONAL:
            checked_basis(self.joint_basis)
            if self.joint_basis['source_split']!='train':raise ValueError('関節座標は学習splitだけで計算してください')
        elif self.joint_basis is not None:raise ValueError('関節座標の指定が不足しています')
        base=policy.get_base_model() if isinstance(policy,PeftModel) else policy
        tokenizer=base.model.vlm_with_expert.processor.tokenizer
        tokens=tokenizer([self.instruction],padding='max_length',max_length=policy.config.tokenizer_max_length,
                         truncation=True,return_tensors='pt')
        self.language={OBS_LANGUAGE_TOKENS:tokens['input_ids'].to('mps'),
                       OBS_LANGUAGE_ATTENTION_MASK:tokens['attention_mask'].to('mps').bool()}

    def normalized(self,key,value):
        value=torch.as_tensor(value,dtype=torch.float32,device='mps')
        stats=self.stats[key]
        return (value-torch.tensor(stats['mean'],device='mps'))/torch.tensor(stats['std'],device='mps')

    def batch(self,sample):
        result={**self.language,'observation.state':self.normalized('observation.state',sample['observation.state']).unsqueeze(0)}
        for i in (1,2):
            key=f'observation.images.camera{i}'
            image=sample[key].to(device='mps',dtype=torch.float32)
            if image.shape!=(3,IMAGE_SIZE,IMAGE_SIZE) or not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
                raise ValueError('RGB画像の契約不一致')
            result[key]=image.unsqueeze(0)
        if 'action' in sample:
            actions=sample['action']
            if self.joint_encoding==RELATIVE:actions=relative_action(actions,sample['observation.state'])
            if self.robot_action_space==CARTESIAN_SPACE:actions=encode_cartesian(actions.detach().cpu().numpy())
            elif self.action_encoding==BOUNDED:actions=encode(actions.detach().cpu().numpy(),BOUNDED)
            if self.joint_encoding==ORTHOGONAL:
                actions=np.asarray(actions,dtype=np.float64).copy()
                actions[...,:7]=encode_joints(actions[...,:7],self.joint_basis)
            result['action']=self.normalized('action',actions).unsqueeze(0)
            result['action_is_pad']=sample['action_is_pad'].to('mps').bool().unsqueeze(0)
        return result

    def denormalize_action(self,action,reference_state=None):
        stats=self.stats['action']
        values=(action.detach().cpu()*torch.tensor(stats['std'])+torch.tensor(stats['mean'])).numpy()
        result=decode_cartesian(values) if self.robot_action_space==CARTESIAN_SPACE else (decode(values,BOUNDED) if self.action_encoding==BOUNDED else values)
        if self.joint_encoding==ORTHOGONAL:
            result=result.astype(np.float64,copy=True)
            result[...,:7]=decode_joints(result[...,:7],self.joint_basis)
        if self.joint_encoding==RELATIVE:result=absolute_action(result,reference_state).numpy()
        return result
