#!/usr/bin/env python
# timing_benchmark.py — wall-clock timing for a single policy (no training, no checkpoint).
#
# Measures, for ONE policy on ONE (untrained) build:
#   - train step:  time for forward + backward + optimizer.step on one batch
#   - inference:   time for one full action-chunk inference (reset + select_action)
#
# It reuses the SAME config plumbing as lerobot.scripts.train (@parser.wrap +
# TrainPipelineConfig), so every `--policy.type=...`, `--dataset.repo_id=...`,
# `--batch_size=...` flag behaves identically to a real training command. The
# policy is built from dataset metadata via make_policy but never trained — we
# only care about per-step latency, so any dataset of the right shape works.
#
# Knobs not covered by TrainPipelineConfig are read from the environment so we
# don't have to fight draccus for extra CLI args:
#   BENCH_LABEL        row label in the CSV          (default: cfg.policy.type)
#   BENCH_CSV          results CSV (appended)        (default: ./timing_results.csv)
#   BENCH_WARMUP       train warmup iters            (default: 5)
#   BENCH_ITERS        train timed iters             (default: 30)
#   BENCH_INFER_WARMUP inference warmup iters        (default: 10)
#   BENCH_INFER_ITERS  inference timed iters         (default: 50)
#   BENCH_INFER_BS     inference batch size          (default: 1 — single robot)
#   BENCH_TASK         language string for lavact    (default: a sorting prompt)
#
# Example (inside lerobot:latest, after datasets are pulled):
#   BENCH_LABEL=concept_act_tce BENCH_CSV=/workspace/timing.csv \
#   python /lerobot/cact_scripts/timing_benchmark.py \
#       --dataset.repo_id=sim/sort_object_with_concepts_cube_green \
#       --policy.type=concept_act --policy.use_concept_learning=true \
#       --policy.concept_method=transformer_ce --policy.use_class_aware_concepts=true \
#       --policy.device=cuda --policy.n_heads=16 --batch_size=32 --steps=1
import csv
import json
import logging
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _synth_observation(policy, device: torch.device, batch_size: int, task: str) -> dict:
    """Build a single-frame observation batch from the policy's input features.

    select_action expects a single timestep (it manages any temporal history
    internally), so we synthesise per-key tensors with no time dimension. Values
    are irrelevant for timing — the model's latency is data-independent. Images
    are in [0, 1] (pixel-like), everything else is gaussian. A `task` string is
    always included (required by lavact, ignored by the others).
    """
    obs: dict = {}
    for key, ft in policy.config.input_features.items():
        shape = (batch_size, *ft.shape)
        if ft.type is FeatureType.VISUAL:
            obs[key] = torch.rand(*shape, device=device)
        else:
            obs[key] = torch.randn(*shape, device=device)
    obs["task"] = [task] * batch_size
    return obs


@parser.wrap()
def benchmark(cfg: TrainPipelineConfig):
    cfg.validate()  # needs --steps or --epochs; pass --steps=1 from the wrapper

    label = os.environ.get("BENCH_LABEL", cfg.policy.type)
    csv_path = Path(os.environ.get("BENCH_CSV", "timing_results.csv"))
    warmup = _env_int("BENCH_WARMUP", 5)
    iters = _env_int("BENCH_ITERS", 30)
    infer_warmup = _env_int("BENCH_INFER_WARMUP", 10)
    infer_iters = _env_int("BENCH_INFER_ITERS", 50)
    infer_bs = _env_int("BENCH_INFER_BS", 1)
    task = os.environ.get("BENCH_TASK", "sort the object into the correct box")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    # Match train.py so timings reflect the real training/inference path.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"[{label}] building dataset + policy ...")
    dataset = make_dataset(cfg)
    ds_meta = dataset._datasets[0].meta if hasattr(dataset, "_datasets") else dataset.meta
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    policy.to(device)

    n_total = sum(p.numel() for p in policy.parameters())
    n_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # One real training batch (with action + concept targets) drives the train step.
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    train_batch = _to_device(next(iter(loader)), device)

    # ── train-step timing: forward + backward + optimizer.step ────────────────
    # A plain AdamW is enough — optimizer.step latency doesn't depend on the
    # per-param-group LR multipliers the real preset uses, and this sidesteps the
    # diffusion scheduler/steps coupling in make_optimizer_and_scheduler.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    policy.train()

    def _train_step():
        optimizer.zero_grad(set_to_none=True)
        out = policy.forward(train_batch)
        loss = out[0] if isinstance(out, (tuple, list)) else out
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        _train_step()
    _sync(device)

    train_times = []
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        out = policy.forward(train_batch)
        loss = out[0] if isinstance(out, (tuple, list)) else out
        loss.backward()
        optimizer.step()
        _sync(device)
        train_times.append(time.perf_counter() - t0)

    # ── inference timing: one full action-chunk per call ──────────────────────
    # reset() before each call empties the action queue so every select_action
    # triggers a full model forward (the real per-decision cost), uniformly across
    # ACT / ConceptACT / LAV-ACT / Diffusion. reset() is outside the timed region.
    policy.eval()
    obs = _synth_observation(policy, device, infer_bs, task)

    for _ in range(infer_warmup):
        policy.reset()
        policy.select_action(obs)
    _sync(device)

    infer_times = []
    for _ in range(infer_iters):
        policy.reset()
        _sync(device)
        t0 = time.perf_counter()
        policy.select_action(obs)
        _sync(device)
        infer_times.append(time.perf_counter() - t0)

    train_ms = [t * 1e3 for t in train_times]
    infer_ms = [t * 1e3 for t in infer_times]
    train_mean = statistics.mean(train_ms)
    infer_mean = statistics.mean(infer_ms)

    row = {
        "label": label,
        "policy_type": cfg.policy.type,
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_batch_size": cfg.batch_size,
        "infer_batch_size": infer_bs,
        "train_ms_mean": round(train_mean, 3),
        "train_ms_std": round(statistics.pstdev(train_ms), 3),
        "train_samples_per_s": round(cfg.batch_size / (train_mean / 1e3), 1),
        "infer_ms_mean": round(infer_mean, 3),
        "infer_ms_std": round(statistics.pstdev(infer_ms), 3),
        "infer_hz": round(infer_bs / (infer_mean / 1e3), 1),
        "n_params_total": n_total,
        "n_params_learnable": n_learnable,
        "train_warmup": warmup,
        "train_iters": iters,
        "infer_warmup": infer_warmup,
        "infer_iters": infer_iters,
        "dataset": cfg.dataset.repo_id if isinstance(cfg.dataset.repo_id, str) else ",".join(cfg.dataset.repo_id),
        "amp": cfg.policy.use_amp,
    }

    logging.info(
        f"[{label}] train: {row['train_ms_mean']}±{row['train_ms_std']} ms/batch "
        f"({row['train_samples_per_s']} samples/s)  |  "
        f"infer: {row['infer_ms_mean']}±{row['infer_ms_std']} ms/call "
        f"({row['infer_hz']} Hz, bs={infer_bs})  |  params={n_total/1e6:.1f}M"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    json_path = csv_path.with_name(f"{label}.json")
    with open(json_path, "w") as f:
        json.dump({**row, "train_ms": train_ms, "infer_ms": infer_ms}, f, indent=2)
    logging.info(f"[{label}] appended → {csv_path}  (detail → {json_path})")


if __name__ == "__main__":
    init_logging()
    benchmark()
