"""教師実演の保存記録を監査する。表面交差の独立再計算ではない。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from .evaluation_reservation import require_development_person

RETREAT = .25
RETREAT_TOLERANCE = .005
RELEASE_LIMIT = .05
OPEN_MINIMUM = .9
IMAGE_SIZE = (256, 256)
LOW = (-.08, -.16, .02, 0)
HIGH = (.08, .12, .35, 3)
ROUNDING_TOLERANCE = 1e-9


def check_outcome(meta, row_count, expected_frames):
    if meta["frames"] != row_count:
        raise ValueError("保存行数と実演記録が一致しません")
    state = meta["final_state"]
    released = state[3] <= RELEASE_LIMIT
    retreated = abs(state[2] - RETREAT) <= RETREAT_TOLERANCE
    if released != meta["released"] or retreated != meta["retreated"]:
        raise ValueError("最終状態と解除・退避記録が一致しません")
    hits = meta["hits"]
    accepted = (not meta["error"] and not meta["contacts"] and
                row_count == expected_frames and meta["drop_count"] == 1 and
                len(hits) == 1 and hits[0]["result"] == "target" and
                hits[0]["openness"] >= OPEN_MINIMUM and released and retreated)
    if accepted != meta["accepted"]:
        raise ValueError("採否が着滴・接触・解除・退避記録と一致しません")
    return accepted


def audit(root, snapshot, source_root):
    protocol = json.loads((root / "protocol.json").read_bytes())
    summary = json.loads((root / "collection-summary.json").read_bytes())
    frozen = json.loads(snapshot.read_bytes())
    for name, digest in frozen["source_files"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"実行ソースが変化しています: {name}")
    for key, value in frozen["protocol"].items():
        if protocol.get(key) != value:
            raise ValueError(f"固定条件と不一致: {key}")
    expected = {(person, split, seed) for person in protocol["people"]
                for split in ("train", "validation") for seed in protocol[f"{split}_seeds"]}
    if len(summary) != len(expected):
        raise ValueError("全条件の収集が完了していません")
    folders = sorted(root.glob("episode-*"))
    if len(folders) != len(expected):
        raise ValueError("実演フォルダー数が一致しません")
    remaining = set(expected)
    records = []
    expected_frames = round(protocol["hz"] * protocol["duration_s"])
    for folder, entry in zip(folders, summary, strict=True):
        meta_bytes = (folder / "metadata.json").read_bytes()
        steps_bytes = (folder / "steps.json").read_bytes()
        meta, rows = json.loads(meta_bytes), json.loads(steps_bytes)
        require_development_person(meta["person"], meta.get("seed"))
        key = (meta["person"], meta["split"], meta["seed"])
        if key not in remaining:
            raise ValueError("人物・条件が重複または固定条件外です")
        remaining.remove(key)
        for field in ("episode", "person", "seed", "split", "accepted", "frames"):
            if meta[field] != entry[field]:
                raise ValueError("一覧と実演記録が一致しません")
        accepted = check_outcome(meta, len(rows), expected_frames)
        images_digest = hashlib.sha256()
        for row in rows:
            if set(row) != {"image", "state", "action"}:
                raise ValueError("未定義の学習入力です")
            for field in ("state", "action"):
                values = row[field]
                if len(values) != len(LOW) or any(not math.isfinite(v) or
                        v < lo - ROUNDING_TOLERANCE or v > hi + ROUNDING_TOLERANCE
                        for v, lo, hi in zip(values, LOW, HIGH)):
                    raise ValueError("状態または操作が範囲外です")
            image = (folder / row["image"]).resolve()
            if not image.is_relative_to(folder.resolve()):
                raise ValueError("画像参照が範囲外です")
            images_digest.update(image.read_bytes())
            with Image.open(image) as opened:
                if opened.size != IMAGE_SIZE or opened.mode not in ("RGB", "RGBA"):
                    raise ValueError("画像の形式が不正です")
                if opened.mode == "RGBA" and opened.getchannel("A").getextrema() != (255, 255):
                    raise ValueError("透明画素が含まれています")
                opened.load()
        records.append(dict(episode=folder.name, person=meta["person"], split=meta["split"],
                            seed=meta["seed"], accepted=accepted, frames=len(rows),
                            images_sha256=images_digest.hexdigest(),
                            metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                            steps_sha256=hashlib.sha256(steps_bytes).hexdigest()))
    return dict(scope="Recorded outcomes and input integrity; not independent geometry replay or learned success",
                episodes=len(records), accepted=sum(r["accepted"] for r in records),
                frames=sum(r["frames"] for r in records), rows=records)


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "snapshot", "source_root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("既存の監査は上書きしません")
    report = audit(args.root, args.snapshot, args.source_root)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
