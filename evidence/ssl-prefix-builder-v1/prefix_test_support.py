"""SYNTHETIC_CORRECTNESS_ONLY private in-memory fixture transport.

Not a filesystem denial workaround: no attempted denied operation is redirected.
Root alone enables optimizer fixtures in its integrated checkout.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def npz(data):
    out = io.BytesIO()
    np.savez_compressed(out, **data)
    return out.getvalue()


def metadata(deployments=("synthetic-site",), cutoffs=(0, 17, 24, 140, 360, 695, 696, 888, 912)):
    sources, intervals, rows = {}, {}, []
    base = datetime.fromisoformat("2026-01-01T00:00:00")
    for dep in deployments:
        source = {
            "deployment": dep,
            "site": dep + "-site",
            "archive": dep + "-archive",
            "configuration": dep + "-config",
            "start": base.isoformat(),
            "end": (base + timedelta(days=50)).isoformat(),
            "native_bounds_m": [0, 230],
            "channel_bounds_m": [[0, 230]] * 4,
        }
        sources[dep] = source
        for hours in cutoffs:
            cutoff = base + timedelta(days=4, hours=hours)
            context, target = [], []
            for offsets, dest in ((range(-95, 1), context), ((1, 3, 6), target)):
                for offset in offsets:
                    stamp = cutoff + timedelta(hours=offset)
                    ids = []
                    for c in range(4) if dest is context else (0,):
                        identity = f"{dep}:{stamp.isoformat()}:c{c}"
                        intervals[identity] = {
                            **{
                                k: source[k]
                                for k in ("deployment", "site", "archive", "configuration")
                            },
                            "timestamp": stamp.isoformat(),
                            "source_interval_index": int((stamp - base).total_seconds() // 3600),
                            "channel": c,
                            "pings": 150,
                            "native_bounds_m": [0, 230],
                        }
                        ids.append(identity)
                    dest.append(ids if dest is context else ids[0])
            rows.append(
                {
                    **{k: source[k] for k in ("deployment", "site", "archive", "configuration")},
                    "row_id": f"SYNTHETIC_CORRECTNESS_ONLY-row-{dep}-{hours}",
                    "cutoff": cutoff.isoformat(),
                    "context_ids": context,
                    "target_ids": target,
                    "target_observed": [True, True, True],
                }
            )
    return {
        "kind": "native_prefix_raw_intervals_v1",
        "complete": True,
        "sources": sources,
        "intervals": intervals,
        "rows": rows,
    }


def arrays(rows, registry, role="prefix", count=None):
    if count is not None:
        rows = [
            {
                "row_id": f"SYNTHETIC_CORRECTNESS_ONLY-dev-{i}",
                "deployment": "synthetic-original-dev",
            }
            for i in range(count)
        ]
    n = len(rows)
    x = np.sin(np.arange(n * 96 * 4).reshape(n, 96, 4) / 71).astype(np.float32) * 5 - 50
    observed = np.ones_like(x, bool)
    observed[:, ::3, 2] = False
    x[~observed] = np.nan
    meta = np.zeros((n, 4, 10), np.float32)
    meta[..., 0] = np.asarray([38000, 125000, 200000, 455000]) / 455000
    meta[..., 1:3] = 1
    meta[..., 4] = (230 if role == "prefix" else 225) / 250
    query = np.repeat(meta[:, :1], 3, axis=1)
    query[..., 9] = [1, 3, 6]
    targets = np.tile(np.asarray([-46, -48, -50], np.float32), (n, 1))
    masks = np.ones_like(targets, bool)
    dates = (
        [[registry["intervals"][v]["timestamp"][:10] for v in r["target_ids"]] for r in rows]
        if role == "prefix"
        else [["2026-05-01"] * 3] * n
    )
    return {
        "x": x,
        "context_observed": observed,
        "metadata": meta,
        "query": query,
        "targets": targets,
        "target_observed": masks,
        "row_id": np.asarray([r["row_id"] for r in rows]),
        "deployment": np.asarray([r["deployment"] for r in rows]),
        "target_dates": np.asarray(dates),
        "corpus_role": np.asarray(role),
        "evidence_kind": np.asarray("SYNTHETIC_CORRECTNESS_ONLY"),
        "fixture_identity": np.asarray("SYNTHETIC_CORRECTNESS_ONLY"),
        "future": np.asarray([{"FORBIDDEN": "suffix/future decode"}], dtype=object),
    }


class MemoryFS:
    def __init__(self, monkeypatch, base):
        self.base, self.files, self.dirs = base, {}, {base}
        old_read, old_exists, old_dir, old_open, old_mkdir = (
            Path.read_bytes,
            Path.exists,
            Path.is_dir,
            Path.open,
            Path.mkdir,
        )

        def owns(p):
            return p == base or p.is_relative_to(base)

        def read(p):
            if owns(p):
                return self.files[p]
            return old_read(p)

        def exists(p):
            return p in self.files or p in self.dirs if owns(p) else old_exists(p)

        def is_dir(p):
            return p in self.dirs if owns(p) else old_dir(p)

        def mkdir(p, *a, **kw):
            if not owns(p):
                return old_mkdir(p, *a, **kw)
            if p in self.dirs and not kw.get("exist_ok", False):
                raise FileExistsError(p)
            self.dirs.add(p)

        fs = self

        class Sink(io.BytesIO):
            def __init__(self, p, text):
                super().__init__()
                self.path, self.text = p, text

            def write(self, v):
                return super().write(v.encode("utf-8") if isinstance(v, str) else v)

            def close(self):
                if not self.closed:
                    fs.files[self.path] = self.getvalue()
                super().close()

        def opened(p, mode="r", *a, **kw):
            if not owns(p):
                return old_open(p, mode, *a, **kw)
            if "x" in mode and exists(p):
                raise FileExistsError(p)
            if mode in ("x", "xb", "wb"):
                return Sink(p, "b" not in mode)
            if "b" in mode:
                return io.BytesIO(read(p))
            return io.StringIO(read(p).decode())

        for key, value in (
            ("read_bytes", read),
            ("exists", exists),
            ("is_dir", is_dir),
            ("mkdir", mkdir),
            ("open", opened),
        ):
            monkeypatch.setattr(Path, key, value)


def case(prefix, monkeypatch, root, mode="scratch_direct"):
    base = root / "evidence/ssl-prefix-builder-v1/SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL"
    fs = MemoryFS(monkeypatch, base)
    config = prefix.PrefixConfig(
        method="direct", mode=mode, correctness_smoke=True, updates=4, cadence=1, batch_size=8
    )
    backbone = prefix.core.Config(
        method="direct", width=8, latent=4, blocks=1, heads=2, seed=7
    ).to_dict()
    registry = metadata(cutoffs=tuple(range(18)) + tuple(range(888, 906)))
    partition = prefix.partitions(registry, 1)
    fit_keys = set(partition["fit_rows"])
    fit_rows = [r for r in registry["rows"] if (r["deployment"], r["row_id"]) in fit_keys]
    train_members = [
        {
            "site": "synthetic-TRAIN-site",
            "deployment": "synthetic-TRAIN-deployment",
            "archive": "synthetic-TRAIN-archive",
        }
    ]
    stats = {
        "channel_mean": [-50.0] * 4,
        "channel_std": [5.0] * 4,
        "target_mean": [-50.0] * 3,
        "target_std": [5.0] * 3,
    }
    documents = {
        "config.json": config.to_dict(),
        "backbone.json": backbone,
        "raw.json": registry,
        "prefix-cohort.json": {"complete": True, "role": "prefix", "rows": partition["fit_rows"]},
        "dev-cohort.json": {
            "complete": True,
            "role": "development",
            "native_bounds_m": [0, 225],
            "rows": [
                ["synthetic-original-dev", f"SYNTHETIC_CORRECTNESS_ONLY-dev-{i}"] for i in range(18)
            ],
        },
        "train-cohort.json": {"role": "train", "complete": True, "members": train_members},
        "stats.json": stats,
        "sources.txt": {"sources": registry["sources"], "quarantined_pings": [165]},
        "split.txt": {
            "sources": [
                *[
                    {"deployment": s["deployment"], "archive_sha256": s["archive"], "role": "test"}
                    for s in registry["sources"].values()
                ],
                {
                    "deployment": "synthetic-original-dev",
                    "archive_sha256": "synthetic-dev-archive",
                    "role": "development",
                },
                {
                    "deployment": "synthetic-TRAIN-deployment",
                    "archive_sha256": "synthetic-TRAIN-archive",
                    "role": "train",
                },
            ]
        },
    }
    manifest = {
        "kind": "native_prefix_transfer_manifest_v1",
        "role": "prefix_transfer",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY",
        "implementer_session_id": prefix.IMPLEMENTER_SESSION_ID,
        "coordinator_session_id": "synthetic-coordinator",
        "config": config.to_dict(),
        "config_path": "config.json",
        "backbone_config": "backbone.json",
        "protocol": "protocol.txt",
        "split": "split.txt",
        "selection": "selection.txt",
        "dependency_lock": "lock.txt",
        "supervisor": "supervisor.txt",
        "source_manifest": "sources.txt",
        "raw_intervals": "raw.json",
        "prefix_cohort": "prefix-cohort.json",
        "dev_cohort": "dev-cohort.json",
        "scalers": "stats.json",
        "scaler_ancestry": "ancestry.json",
        "encoder": None,
        "prefix_npz": "prefix.npz",
        "dev_npz": "dev.npz",
        "numeric_access_review": "numeric.json",
    }
    for name in (
        "protocol.txt",
        "split.txt",
        "selection.txt",
        "lock.txt",
        "supervisor.txt",
        "sources.txt",
        "train.npz",
        "train-intervals.json",
    ):
        fs.files[base / name] = b"SYNTHETIC_CORRECTNESS_ONLY never decoded ancestor/source bytes"
    node = {
        "kind": "native_prefix_train_ancestry_v1",
        "complete": True,
        "role": "train",
        "historical_initial_weights": False,
        "parents": [],
        "inputs": [
            {
                "cohort": "train-cohort.json",
                "npz": "train.npz",
                "raw_intervals": "train-intervals.json",
                "members": train_members,
            }
        ],
        "artifacts": [{"path": "stats.json", "sha256": prefix.sha(encoded(stats))}],
    }
    documents["ancestry.json"] = node
    documents["selection.txt"] = {
        "role": "original_development",
        "cohort_sha256": prefix.sha(encoded(documents["dev-cohort.json"])),
        "opportunities": [1, 2, 3, 4],
        "floor": 18,
        "aggregation": "equal_target_source_date_then_horizon_then_deployment",
    }
    documents["train-intervals.json"] = {
        "kind": "native_prefix_train_interval_receipt_v1",
        "role": "train",
        "complete": True,
        "members": train_members,
        "source_npz_sha256": prefix.sha(fs.files[base / "train.npz"]),
        "cohort_sha256": prefix.sha(encoded(documents["train-cohort.json"])),
        "split_sha256": prefix.sha(encoded(documents["split.txt"])),
    }
    holder = SimpleNamespace(
        fs=fs,
        base=base,
        config=config,
        backbone=backbone,
        registry=registry,
        documents=documents,
        manifest=manifest,
        output=base / "out",
        manifest_path=base / "manifest.json",
        review_path=base / "review.json",
        prefix_arrays=arrays(fit_rows, registry),
        dev_arrays=arrays([], registry, "development", count=18),
    )

    def seal():
        for name, value in documents.items():
            fs.files[base / name] = encoded(value)
        fs.files[base / "prefix.npz"] = npz(holder.prefix_arrays)
        fs.files[base / "dev.npz"] = npz(holder.dev_arrays)
        fs.files[holder.manifest_path] = encoded(manifest)
        bindings = {str(p): prefix.sha(p.read_bytes()) for p in prefix.required_sources()}
        bindings.update(
            {
                str(p): prefix.sha(raw)
                for p, raw in fs.files.items()
                if p.name not in ("numeric.json", "review.json")
            }
        )
        common = {
            "role": "prefix_transfer",
            "evidence_kind": manifest["evidence_kind"],
            "implementer_session_id": manifest["implementer_session_id"],
            "coordinator_session_id": manifest["coordinator_session_id"],
            "reviewer_session_id": "synthetic-distinct-reviewer",
            "config": manifest["config"],
        }
        numeric = {
            **common,
            "status": "APPROVED_PREFIX_TRANSFER_NUMERIC_ACCESS",
            "scope": "prefix_only_numeric_fit",
            "bindings": dict(bindings),
            "allowed_uses": [
                "prefix_fit",
                "original_development_selection",
                "suffix_metadata_only",
            ],
        }
        fs.files[base / "numeric.json"] = encoded(numeric)
        bindings[str(base / "numeric.json")] = prefix.sha(fs.files[base / "numeric.json"])
        review = {
            **common,
            "status": "APPROVED_PREFIX_TRANSFER_PREFIT",
            "scope": "native_prefix_transfer_fit",
            "bindings": bindings,
            "allowed_cells": [
                {
                    k: manifest["config"][k]
                    for k in ("method", "mode", "seed", "prefix_days", "family")
                }
            ],
        }
        fs.files[holder.review_path] = encoded(review)
        holder.review, holder.numeric = review, numeric

    holder.seal = seal
    seal()
    return holder


def frozen_case(prefix, monkeypatch, root):
    holder = case(prefix, monkeypatch, root, mode="frozen_readout")
    config = holder.config
    scratch = prefix.PrefixConfig(**{**config.to_dict(), "mode": "scratch_direct"})
    model = prefix.prepare_model(scratch, holder.backbone, None, holder.documents["stats.json"])
    bindings = {str(p): prefix.sha(p.read_bytes()) for p in prefix.required_sources()}
    selected = {
        "kind": "native_ssl_selected_encoder_v1",
        "config": holder.backbone,
        "scalers": holder.documents["stats.json"],
        "bindings": bindings,
        "encoder": prefix.core.cpu_state(model.encoder),
    }
    inference = {
        "kind": "native_ssl_weights_only_inference_v1",
        "config": holder.backbone,
        "scalers": holder.documents["stats.json"],
        "bindings": bindings,
        "model": {"encoder." + k: v for k, v in selected["encoder"].items()},
    }
    holder.fs.files[holder.base / "selected.pt"] = prefix.encode_checkpoint(selected)
    holder.fs.files[holder.base / "parent-inference.pt"] = prefix.encode_checkpoint(inference)
    holder.fs.files[holder.base / "membership.json"] = encoded({"complete": True, "role": "train"})
    holder.documents["parent-run.json"] = {
        "config": holder.backbone,
        "bindings": bindings,
        "inference_sha256": prefix.sha(holder.fs.files[holder.base / "parent-inference.pt"]),
        "membership_sha256": prefix.sha(holder.fs.files[holder.base / "membership.json"]),
    }
    holder.manifest.update(
        encoder="selected.pt",
        encoder_ancestry="ancestry.json",
        parent_run="parent-run.json",
        parent_inference="parent-inference.pt",
        parent_membership="membership.json",
    )
    for name in ("selected.pt", "parent-inference.pt", "membership.json"):
        holder.documents["ancestry.json"]["artifacts"].append(
            {"path": name, "sha256": prefix.sha(holder.fs.files[holder.base / name])}
        )
    holder.selected = selected
    holder.seal()
    return holder
