"""教師と推論に共通する指幅の表現。物理指令の事後補正には使わない。"""
import numpy as np

LINEAR='linear-v1'
BOUNDED='bounded-gap-logit-v1'
LOWER_M=.0129
UPPER_M=.08
ENDPOINT_FRACTION=1e-6
INPUT_ROUNDING_M=1e-8
ACTION_DIM=8


def checked(values,mode):
    if mode not in (LINEAR,BOUNDED):raise ValueError('未対応の動作表現です')
    values=np.asarray(values)
    if values.ndim<1 or values.shape[-1]!=ACTION_DIM or not np.isfinite(values).all():
        raise ValueError('動作の形または値が不正です')
    return values


def encode(values,mode):
    values=checked(values,mode)
    if mode==LINEAR:return values.copy()
    result=values.astype(np.float64,copy=True)
    result[...,-1]=encode_gap(result[...,-1])
    return result


def encode_gap(gap):
    gap=np.asarray(gap,dtype=np.float64)
    if not np.isfinite(gap).all():raise ValueError('指幅が非有限です')
    if np.any(gap<LOWER_M-INPUT_ROUNDING_M) or np.any(gap>UPPER_M+INPUT_ROUNDING_M):
        raise ValueError('教師の指幅が物理範囲外です')
    fraction=(gap-LOWER_M)/(UPPER_M-LOWER_M)
    # float32の端点丸めとlogitの無限大だけを教師側で有限符号化する。
    fraction=np.where(np.abs(gap-LOWER_M)<=INPUT_ROUNDING_M,ENDPOINT_FRACTION,fraction)
    fraction=np.where(np.abs(gap-UPPER_M)<=INPUT_ROUNDING_M,1-ENDPOINT_FRACTION,fraction)
    return np.log(fraction)-np.log1p(-fraction)


def decode(values,mode):
    values=checked(values,mode)
    if mode==LINEAR:return values.copy()
    result=values.astype(np.float64,copy=True)
    result[...,-1]=decode_gap(result[...,-1])
    return result


def decode_gap(latent):
    latent=np.asarray(latent,dtype=np.float64)
    if not np.isfinite(latent).all():raise ValueError('指幅の潜在値が非有限です')
    return LOWER_M+(UPPER_M-LOWER_M)*np.exp(-np.logaddexp(0.,-latent))
