"""予約人物を除外し、改善用へ移行した人物も新規seedだけを学習へ許可する。"""
RESERVED_EVALUATION_PEOPLE = frozenset({
    "alicia", "tsukuyomi",
})

# B/Cは固定Cでの評価を保存後、2026-10-22に改善用へ移行した。
# 以後の成績を未使用人物への汎化と呼ばない。履歴は人物移行記録に保持する。
ADAPTED_EVALUATED_PEOPLE = frozenset({"avatar-sample-b", "avatar-sample-c"})
ADAPTATION_SEEDS = frozenset(range(14001, 14017)) | {14101, 14102}


def require_development_person(person, seed=None):
    if person in RESERVED_EVALUATION_PEOPLE:
        raise ValueError("最終評価用に予約した人物は学習へ使用できません")
    if person in ADAPTED_EVALUATED_PEOPLE and seed not in ADAPTATION_SEEDS:
        raise ValueError("最終評価の履歴を持つ人物の未許可seedは学習へ使用できません")
