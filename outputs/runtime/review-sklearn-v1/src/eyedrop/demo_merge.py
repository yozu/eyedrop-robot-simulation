"""採用済み実演を重複なく統合し、元ファイルとの対応を保存する。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .evaluation_reservation import require_development_person

PROTOCOL_FIELDS = (
    "scenario", "train_seeds", "validation_seeds", "test_seeds",
    "hz", "duration_s", "approach", "time_budgeted_teacher",
)


def read_json(path):
    return json.loads(path.read_bytes())


def merge_demos(sources, output):
    output = Path(output)
    if output.exists():
        raise ValueError("既存データは上書きしません")
    records, seen, people = [], set(), []
    common = None
    # 全入力の検査を終えるまで出力を作らない。
    for source in map(Path, sources):
        protocol = read_json(source / "protocol.json")
        current = {key: protocol[key] for key in PROTOCOL_FIELDS}
        if common is None:
            common = current
        elif common != current:
            raise ValueError("収集条件・分割が異なります")
        for folder in sorted(source.glob("episode-*")):
            meta = read_json(folder / "metadata.json")
            person, split, seed = meta["person"], meta["split"], meta["seed"]
            require_development_person(person, meta.get("seed"))
            if person not in protocol["people"]:
                raise ValueError("収集対象にない人物です")
            if split not in ("train", "validation") or seed not in protocol[f"{split}_seeds"] or seed in protocol["test_seeds"]:
                raise ValueError("固定した分割に一致しません")
            if meta.get("accepted") is not True or meta["contacts"]:
                raise ValueError("不採用・接触した実演は統合できません")
            key = (person, seed)
            if key in seen:
                raise ValueError("同じ人物・条件が重複しています")
            seen.add(key)
            if person not in people:
                people.append(person)
            files = {"metadata.json", "steps.json"}
            for row in read_json(folder / "steps.json"):
                if set(row) != {"image", "state", "action"}:
                    raise ValueError("学習列に未定義の入力があります")
                image = (folder / row["image"]).resolve()
                if not image.is_relative_to(folder.resolve()) or image.suffix != ".png":
                    raise ValueError("画像参照が不正です")
                files.add(row["image"])
            hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                      for name in sorted(files)}
            records.append(dict(source=str(source.resolve()), source_episode=folder.name,
                                person=person, split=split, seed=seed, files=hashes))
    if not records:
        raise ValueError("統合する実演がありません")
    output.mkdir(parents=True)
    for index, record in enumerate(records):
        record["episode"] = f"episode-{index:03d}"
        destination = output / record["episode"]
        destination.mkdir()
        for name, digest in record["files"].items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(record["source"]) / record["source_episode"] / name, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("コピー中に原本が変化しました")
    merged_protocol = dict(common, version="merged-development-demos-v1", people=people)
    report = dict(scope="Accepted development demos only; oracle files excluded", episodes=records)
    for name, content in (("protocol.json", merged_protocol), ("merge.json", report)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    report = merge_demos(args.sources, args.output)
    print(json.dumps({"episodes": len(report["episodes"])}))


if __name__ == "__main__":
    main()
