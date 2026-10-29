"""合成入力で事前学習済みSmolVLAのMac追加学習経路を確認する。"""
import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
IMAGE_SIZE = 512
STATE_DIM = 32
ACTION_DIM = 8
CHUNK_SIZE = 50
FIRST_GATE_STEPS = 10
TOTAL_STEPS = 100
TIME_BUDGET_S = 30 * 60
MEMORY_LIMIT_BYTES = 48 * 1024**3
SEED = 714
LORA_RANK = 8
LEARNING_RATE = 1e-4
CPU_THREADS = 2


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output):
    if output.exists():
        raise FileExistsError(f"出力を上書きしません: {output}")
    output.mkdir(parents=True)
    record = {"stage": "G0a", "input_kind": "synthetic_only", "task_success_evaluated": False,
              "status": "preparing", "model": MODEL_ID, "model_revision": MODEL_REVISION,
              "backbone_revision": VLM_REVISION, "updates": [], "checkpoints": [],
              "windows_training": False, "full_training_started": False}
    record_path = output / "result.json"
    write_json(record_path, record)
    started = time.monotonic()
    try:
        import psutil
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        from peft import PeftModel

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPSが利用できません。CPUへ代替しません")
        torch.set_num_threads(CPU_THREADS)
        torch.manual_seed(SEED)
        torch.mps.manual_seed(SEED)
        record["versions"] = {name: importlib.metadata.version(name)
                              for name in ("torch", "lerobot", "transformers", "peft", "numpy")}
        base_dir = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                         allow_patterns=["config.json", "model.safetensors"], max_workers=1))
        vlm_dir = Path(snapshot_download(VLM_ID, revision=VLM_REVISION,
                                        allow_patterns=["*.json", "*.jinja", "*.jinja2", "*.txt", "tokenizer.model"],
                                        max_workers=1))
        record["preparation_seconds"] = time.monotonic() - started
        record["base_weight_sha256"] = file_hash(base_dir / "model.safetensors")
        compute_started = time.monotonic()
        swap_initial = psutil.swap_memory().used
        process = psutil.Process()

        def load_base():
            cfg = SmolVLAConfig.from_pretrained(base_dir)
            cfg.device = "cpu"
            cfg.use_amp = False
            cfg.push_to_hub = False
            cfg.load_vlm_weights = False
            cfg.vlm_model_name = str(vlm_dir)
            cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
                                  **{f"observation.images.camera{i}": PolicyFeature(
                                      type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)) for i in (1, 2)}}
            cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
            cfg.n_action_steps = 1
            cfg.pretrained_path = base_dir
            policy = SmolVLAPolicy.from_pretrained(base_dir, config=cfg, strict=True, local_files_only=True)
            policy.config.load_vlm_weights = True
            policy.config.device = "mps"
            return policy.to(device="mps", dtype=torch.float32)

        policy = load_base()
        policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": LORA_RANK,
                                                          "lora_alpha": LORA_RANK})
        trainable = {name: value for name, value in policy.named_parameters() if value.requires_grad}
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError("LoRAだけを更新する契約と学習可能パラメータが一致しません")
        record["trainable_parameters"] = sum(value.numel() for value in trainable.values())
        record["total_parameters"] = sum(value.numel() for value in policy.parameters())
        record["trainable_names"] = list(trainable)
        record["input_features"] = {name: dataclasses.asdict(value) for name, value in policy.config.input_features.items()}
        record["schema"] = {"state": STATE_DIM, "action": ACTION_DIM, "cameras": 2, "chunk": CHUNK_SIZE}
        generator = torch.Generator().manual_seed(SEED)
        batch = {f"observation.images.camera{i}": torch.rand(
            1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator).to("mps") for i in (1, 2)}
        batch["observation.state"] = torch.randn(1, STATE_DIM, generator=generator).to("mps")
        batch["action"] = torch.randn(1, CHUNK_SIZE, ACTION_DIM, generator=generator).to("mps") * 0.1
        tokenizer = policy.get_base_model().model.vlm_with_expert.processor.tokenizer
        tokens = tokenizer(["Place one drop in the right eye, release and retreat.\n"],
                           padding="max_length", max_length=policy.config.tokenizer_max_length,
                           truncation=True, return_tensors="pt")
        batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to("mps")
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].to("mps").bool()
        noise = torch.randn(1, CHUNK_SIZE, STATE_DIM, generator=generator).to("mps")
        flow_time = torch.full((1,), 0.5, device="mps")
        optimizer = torch.optim.AdamW(trainable.values(), lr=LEARNING_RATE, fused=False)
        initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
        record["status"] = "running"
        record["learning_rate"] = LEARNING_RATE
        record["optimizer"] = "AdamW, no AMP, no fused, no compile, batch1"
        write_json(record_path, record)

        def resources():
            rss = process.memory_info().rss
            driver = torch.mps.driver_allocated_memory()
            if rss + driver > MEMORY_LIMIT_BYTES:
                raise RuntimeError("RSSとMPS driverの保守的な合計が48GiBを超えました")
            if time.monotonic() - compute_started > TIME_BUDGET_S:
                raise TimeoutError("G0aの計算30分を超えました")
            swap = psutil.swap_memory().used - swap_initial
            if swap > 256 * 1024**2:
                raise RuntimeError("確認開始後のシステムswap使用量が256MiB以上増えました")
            return {"rss_bytes": rss, "mps_driver_bytes": driver,
                    "mps_allocated_bytes": torch.mps.current_allocated_memory(), "swap_delta_bytes": swap}

        for step in range(1, TOTAL_STEPS + 1):
            step_started = time.monotonic()
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.forward(batch, noise=noise, time=flow_time)
            if not torch.isfinite(loss):
                raise RuntimeError("lossが非有限です")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable.values(), 10.0)
            if not torch.isfinite(grad) or grad <= 0:
                raise RuntimeError("勾配が非有限または0です")
            optimizer.step()
            torch.mps.synchronize()
            row = {"step": step, "loss": loss.item(), "grad_norm": grad.item(),
                   "seconds": time.monotonic() - step_started, **resources()}
            record["updates"].append(row)
            if step in (FIRST_GATE_STEPS, TOTAL_STEPS):
                changed = sum(not torch.equal(initial[name], value.detach().cpu())
                              for name, value in trainable.items())
                if not changed:
                    raise RuntimeError("学習可能な重みに変化がありません")
                policy.eval()
                with torch.no_grad():
                    before = policy.predict_action_chunk(batch, noise=noise).detach().cpu()
                if before.shape != (1, CHUNK_SIZE, ACTION_DIM) or not torch.isfinite(before).all():
                    raise RuntimeError("推論結果が出力契約と一致しません")
                checkpoint = output / f"adapter-{step}"
                policy.save_pretrained(checkpoint, safe_serialization=True)
                fresh = load_base()
                restored = PeftModel.from_pretrained(fresh, checkpoint).to("mps").eval()
                with torch.no_grad():
                    after = restored.predict_action_chunk(batch, noise=noise).detach().cpu()
                difference = (before - after).abs().max().item()
                if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
                    raise RuntimeError(f"保存再読込で推論が一致しません: {difference}")
                record["checkpoints"].append({"step": step, "changed_tensors": changed,
                    "reload_max_abs_difference": difference,
                    "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
                    "resources": resources()})
                del restored, fresh, after, before
                gc.collect()
                torch.mps.empty_cache()
                print(f"GATE {step}: finite gradients, weights changed, reload passed", flush=True)
            if step % FIRST_GATE_STEPS == 0:
                write_json(record_path, record)
                print(f"STEP {step}: loss={row['loss']:.6f} seconds={row['seconds']:.3f}", flush=True)
        policy.eval()
        timings = []
        for _ in range(5):
            start = time.monotonic()
            with torch.no_grad():
                prediction = policy.predict_action_chunk(batch, noise=noise)
            torch.mps.synchronize()
            timings.append(time.monotonic() - start)
        record["inference_seconds"] = timings
        record["compute_seconds"] = time.monotonic() - compute_started
        record["status"] = "passed_synthetic_only"
        record["next_gate"] = "G1 mechanism and teacher validation, then G0b real-data validation"
        write_json(record_path, record)
        print(json.dumps({key: record[key] for key in ("status", "compute_seconds", "next_gate")}), flush=True)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        write_json(record_path, record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.output)
