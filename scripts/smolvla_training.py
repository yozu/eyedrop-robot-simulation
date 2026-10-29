"""実行する先頭動作を重視した損失。均等重みでは公式の式と照合する。"""
import torch
import math
from peft import PeftModel
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK


def repeat_observation(batch,count):
    if not isinstance(count,int) or isinstance(count,bool) or count<1:raise ValueError('複製数が不正です')
    if any(not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[0]!=1 for value in batch.values()):
        raise ValueError('単一観測のテンソルだけを複製できます')
    return {key:value.repeat((count,)+(1,)*(value.ndim-1)) for key,value in batch.items()}


def concatenate_observations(batches):
    if not batches:raise ValueError('結合する観測がありません')
    reference=batches[0];keys=set(reference)
    for batch in batches:
        if set(batch)!=keys:raise ValueError('観測の列が一致しません')
        for key,value in batch.items():
            expected=reference[key]
            if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[0]!=1
                or value.shape!=expected.shape or value.dtype!=expected.dtype or value.device!=expected.device):
                raise ValueError('観測の形・型・deviceが一致しません')
    return {key:torch.cat([batch[key] for batch in batches],dim=0) for key in reference}


def sequence_loss(policy,batch,first_weight,noise=None,time=None,gap_weight=1.):
    if not 0<first_weight<=1:raise ValueError('先頭重みが不正です')
    if not math.isfinite(gap_weight) or gap_weight<=0:raise ValueError('指幅重みが不正です')
    base=policy.get_base_model() if isinstance(policy,PeftModel) else policy
    if base.config.adapt_to_pi_aloha:raise ValueError('この実験はPandaの絶対関節指令専用です')
    images,masks=base.prepare_images(batch)
    state=base.prepare_state(batch)
    action=base.prepare_action(batch)
    losses=base.model.forward(images,masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK],state,action,noise,time)
    losses=losses[:,:,:base.config.action_feature.shape[0]]
    steps=losses.shape[1]
    if steps==1 and first_weight!=1:raise ValueError('1動作の学習は先頭重み1で指定してください')
    weights=torch.ones((steps,),device=losses.device) if steps==1 else torch.full((steps,),(1-first_weight)/(steps-1),device=losses.device)
    weights[0]=first_weight
    valid=(~batch['action_is_pad']).to(losses.dtype)
    weights=valid*weights.unsqueeze(0)
    dimensions=torch.ones(losses.shape[-1],device=losses.device,dtype=losses.dtype)
    dimensions[-1]=gap_weight
    return (losses*weights.unsqueeze(-1)*dimensions).sum()/(weights.sum()*dimensions.sum()).clamp_min(1e-8)
