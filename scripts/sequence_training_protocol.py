"""0更新の接続記録を引き継ぎ、学習時の項目だけを更新する。"""
FIXED_SOURCE_KEYS = ('initial_model_sha256', 'initial_rng_sha256', 'manifest', 'dataset', 'micro')


def training_protocol(source, **settings):
    for key in FIXED_SOURCE_KEYS:
        if key in settings and settings[key] != source[key]:
            raise ValueError('接続確認済みの入力を変更できません: ' + key)
    return {**source, **settings}
