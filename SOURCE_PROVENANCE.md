# コードの由来と不足

このリポジトリのコードは、実機や人に使用するものではない。実験当時のソース一覧に保存された130ファイルは、保存元のSHA-256と一致する。実行後に保存・集計へ加えた変更は `source-deltas/` に分けた。

旧候補には、教師を収集するプログラムが参照する一部のコードが欠けていた。次の4ファイルは、3人物それぞれの教師収集時に保存されたソースが一致したため、保存版を加えた。

| ファイル | SHA-256 |
| --- | --- |
| `scripts/pressure_interval_audit.py` | `288e124df3094ebc787ef1a0b88b7b898dc9889f3be28af5ad842a12c603b849` |
| `scripts/panda_eval_cases.py` | `9d0fa287386131306df6bd5015fe4bfbf1257e0a4428d4cfd70a1b7818789754` |
| `scripts/panda_trace_quality.py` | `d1548f01eb99857386e644ad4776c0868d8d5e73852123d838edf26dd6584ecc` |
| `scripts/image_motion_validity.py` | `a57933fc31bd103c49e4baedbbcfee297ab5bb09083ea0319f9f447ea7216173` |

さらに、`action_encoding.py`、`cartesian_action.py`、`direct_gripper.py`、`endpoint_policy.py`、`joint_action_basis.py`、`panda_optimizer.py`、`train_recovered_smolvla.py`、`tree_gripper.py`、`reference_strict95.py` を補った。これらは実験日前の履歴と現在の作業版が一致したものだが、収集時のスナップショットには含まれておらず、実行時の版を確定したものではない。

構文解析で確認した範囲では、`scripts/` のPythonが直接importするプロジェクト内の同名スクリプトは揃った。ただし動的import、データ・設定、外部パッケージ、起動後の振る舞いを網羅する確認ではない。保存した教師の画像、位置認識器、初期・最終重み、乱数状態は含まれていない。再現するための公開方法・容量・ライセンスは未確認で、結果を再実行できると主張しない。
