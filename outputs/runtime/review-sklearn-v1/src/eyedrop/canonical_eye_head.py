"""診断用の閉眼位置回帰器を、固定認識重みとの整合を確認して読む。"""
import hashlib
import json
from pathlib import Path
from .model_integrity import restore_sklearn_model


def read_canonical_regressor(folder,encoder_sha256):
    folder=Path(folder);card=json.loads((folder/'inference.json').read_bytes())
    if card.get('format')!='panda-canonical-ridge-v1' or card.get('fit_calls')!=1:
        raise ValueError('回帰器の保存形式または学習回数が不正です')
    if card.get('encoder_sha256')!=encoder_sha256:
        raise ValueError('固定した認識重みが一致しません')
    raw=(folder/'diagnostic-model.pkl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=card.get('checkpoint_sha256'):
        raise ValueError('回帰器のハッシュが一致しません')
    model=restore_sklearn_model(raw)
    if not callable(getattr(model,'predict',None)):
        raise ValueError('回帰器の予測機能がありません')
    return model
