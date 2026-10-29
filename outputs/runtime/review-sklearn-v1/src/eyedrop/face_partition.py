"""顔認識の人物単位の除外条件を初期モデルまで維持する。"""


def require_matching_partition(report, holdout_person):
    if report.get("holdout_person") != holdout_person:
        raise ValueError("初期モデルと人物除外条件が一致しません")
