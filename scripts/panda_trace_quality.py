"""旧15秒記録と明示した30秒診断の物理時点数を区別する。"""
ACTION_HZ=20
PHYSICS_HZ=240
LEGACY_FRAMES=300
SUPPORTED_FRAMES=(LEGACY_FRAMES,2*LEGACY_FRAMES,4*LEGACY_FRAMES)


def expected_physics_samples(protocol):
    # 旧記録はframes欄がなく、固定300動作だった契約を引き継ぐ。
    frames=protocol.get('frames',LEGACY_FRAMES)
    if type(frames) is not int or frames not in SUPPORTED_FRAMES:raise ValueError('記録時間が未対応または不明確です')
    return frames*PHYSICS_HZ//ACTION_HZ+1
