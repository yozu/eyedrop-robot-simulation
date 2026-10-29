"""信頼済みの保存モデルを、固定した内容と照合してから復元へ渡す。"""
import hashlib
import pickle
from pathlib import Path
import warnings

FIXED_RGB_MODELS = {
    'separate-v2/openness/weights.pt': '7e4c8716130ed4323ab4e6b7ca7e1841d0fb13c4e5f0e89e595b6fe1ac1121a2',
    'target-readout-v1/weights.pt': '0fbf0cf6f8c27bf2cee55920b7e23c15e284883fd38068ae536c7fa5db5e9f54',
    'rgb-depth-v1/position.pkl': 'af7875d1661484823711bf601f68259fe36aca1293abafd604ed69503d2257c1',
    'c-open-v7/calibration.pkl': '0c227f48a987e6233a2fb9760cbaa88f29c901091cef6571ef556ce5cb99a966',
}


def verified_model_bytes(path, expected_sha256):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('保存モデルのハッシュが一致しません: ' + str(path))
    return content


def restore_sklearn_model(content):
    """照合済みpickleの版違いを、予測に使う前に拒否する。"""
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        try:
            return pickle.loads(content)
        except InconsistentVersionWarning as error:
            raise ValueError(
                'scikit-learnの保存版と実行版が異なります: '
                f'{error.original_sklearn_version} / {error.current_sklearn_version}'
            ) from error
