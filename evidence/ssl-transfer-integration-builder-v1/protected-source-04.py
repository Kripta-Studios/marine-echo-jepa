"""SYNTHETIC_CORRECTNESS_ONLY private in-memory fixture transport.

Not a filesystem denial workaround: no attempted denied operation is redirected.
Root alone enables optimizer fixtures in its integrated checkout.
"""

from __future__ import annotations

import copy
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
        "configuration_mode": "legacy_single_configuration_v1",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
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
    masks = (
        np.asarray([r["target_observed"] for r in rows], bool)
        if role == "prefix"
        else np.ones_like(targets, bool)
    )
    dates = (
        [
            [
                registry["intervals"][v]["timestamp"][:10] if isinstance(v, str) else ""
                for v in r["target_ids"]
            ]
            for r in rows
        ]
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
    base = root / "evidence/ssl-prefix-native-config-builder-v3/SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL"
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
        for path in documents.get("parent-run.json", {}).get("bindings", {}):
            actual = Path(path)
            bindings[str(actual)] = prefix.sha(actual.read_bytes())
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


def absent_target(registry, row, horizon_index, reason="SOURCE_GAP"):
    """Declare structural absence from metadata, never infer a missing timestamp."""
    identity = row["target_ids"][horizon_index]
    record = registry["intervals"][identity]
    ref = row["row_id"] + ":absence:" + str(horizon_index)
    row["target_ids"][horizon_index] = -1
    row["target_observed"][horizon_index] = False
    row.setdefault("target_absence_refs", [None] * 3)[horizon_index] = ref
    used = {v for r in registry["rows"] for slots in r["context_ids"] for v in slots}
    used.update(v for r in registry["rows"] for v in r["target_ids"] if isinstance(v, str))
    if identity not in used:
        registry["intervals"].pop(identity)
    gap = {
        "complete": True,
        **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
        "reason": reason,
        "first_source_interval_index": record["source_interval_index"],
        "last_source_interval_index": record["source_interval_index"],
        "source_metadata_sha256": "a" * 64,
    }
    if reason == "PING_QUARANTINE":
        gap["quarantined_pings"] = [165]
    if reason == "CONFIGURATION_BOUNDARY":
        gap["next_configuration"] = "synthetic-next-native-config"
    registry.setdefault("source_gaps", {})[ref] = gap
    return ref


def bound_gaps(holder, prefix):
    evidence = {"kind": "SYNTHETIC_CORRECTNESS_ONLY_SOURCE_METADATA", "reader_gap_fixture": True}
    digest = prefix.sha(encoded(evidence))
    for gap in holder.registry["source_gaps"].values():
        gap["source_metadata_sha256"] = digest
    holder.documents["gap-source-metadata.json"] = evidence
    holder.documents["source-gaps.json"] = {
        "kind": "native_prefix_source_gaps_v1",
        "complete": True,
        "gaps": holder.registry["source_gaps"],
        "source_metadata": {
            ref: "gap-source-metadata.json" for ref in holder.registry["source_gaps"]
        },
    }
    holder.manifest["source_gaps"] = "source-gaps.json"


def refresh_parent_artifacts(holder, prefix):
    """Re-admit intentionally altered private tensors for semantic guard tests."""
    for name, value in (
        ("selected.pt", holder.selected),
        ("parent-inference.pt", holder.parent_inference),
    ):
        raw = prefix.encode_checkpoint(value)
        holder.fs.files[holder.base / name] = raw
        for item in holder.documents["ancestry.json"]["artifacts"]:
            if item["path"] == name:
                item["sha256"] = prefix.sha(raw)
        if name == "selected.pt":
            holder.documents["parent-run.json"]["selected_encoder_sha256"] = prefix.sha(raw)
        else:
            holder.documents["parent-run.json"]["inference_sha256"] = prefix.sha(raw)
    holder.seal()


def native_configuration_metadata(cutoffs=(0, 17, 140, 888, 912)):
    """Metadata-only synthetic source with genuine separate nominal/actual IDs."""
    registry = metadata(cutoffs=cutoffs)
    dep, source = next(iter(registry["sources"].items()))
    source.pop("configuration")
    source.pop("channel_bounds_m")
    configurations = {}
    for pings in (150, 165, 180):
        configurations[f"native-{pings}"] = {
            "channel_bounds_m": [[0, 230], [0, 225 if pings == 180 else 230], [0, 230], [0, 230]],
            "processing_id_or_unknown": [f"native-process-{pings}"] * 4,
            "ping_counts": [pings] * 4,
            "frequency_hz": [38000, 125000, 200000, 455000],
            "interval_seconds": 3600,
        }
    segments = [
        {
            "first_source_interval_index": 0,
            "last_source_interval_index": 119,
            "configuration": "native-150",
        },
        {
            "first_source_interval_index": 120,
            "last_source_interval_index": 130,
            "configuration": "native-165",
        },
        {
            "first_source_interval_index": 131,
            "last_source_interval_index": 1199,
            "configuration": "native-180",
        },
    ]
    source_evidence = {
        "kind": "native_prefix_source_configuration_metadata_v1",
        "complete": True,
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "source": copy.deepcopy(source),
        "configurations": copy.deepcopy(configurations),
        "segments": copy.deepcopy(segments),
    }
    import hashlib

    source_digest = hashlib.sha256(encoded(source_evidence)).hexdigest()
    catalog = {
        "kind": "native_prefix_source_configuration_map_v1",
        "complete": True,
        "quarantined_pings": [165],
        "sources": {
            dep: {
                **source,
                "first_source_interval_index": 0,
                "last_source_interval_index": 1199,
                "configurations": configurations,
                "segments": segments,
                "source_metadata_sha256": source_digest,
            }
        },
        "source_metadata": {dep: "native-source-metadata.json"},
    }
    registry.update(
        kind="native_prefix_raw_intervals_v2",
        configuration_mode="per_source_configuration_map_v1",
        configuration_map=catalog,
    )
    base = datetime.fromisoformat(source["start"])

    def config_at(index):
        return "native-150" if index <= 119 else "native-165" if index <= 130 else "native-180"

    def actual(index, channel):
        stamp = base + timedelta(hours=index)  # Synthetic fixture clock only.
        identity = f"{dep}:{stamp.isoformat()}:c{channel}"
        name = config_at(index)
        cfg = configurations[name]
        registry["intervals"][identity] = {
            **{k: source[k] for k in ("deployment", "site", "archive")},
            "configuration": name,
            "timestamp": stamp.isoformat(),
            "source_interval_index": index,
            "channel": channel,
            "pings": cfg["ping_counts"][channel],
            "processing_id_or_unknown": cfg["processing_id_or_unknown"][channel],
            "native_bounds_m": cfg["channel_bounds_m"][channel],
            "observed": cfg["ping_counts"][channel] in (150, 180),
            "qc": "OBSERVED_CENSORING_UNKNOWN"
            if name != "native-165"
            else "PARTIAL_SOURCE_INTERVAL",
        }
        return identity

    for row in registry["rows"]:
        cutoff_index = int((datetime.fromisoformat(row["cutoff"]) - base).total_seconds() // 3600)
        name = config_at(cutoff_index)
        row["configuration"] = name
        row["context_ids"] = [
            [actual(i, c) for c in range(4)] for i in range(cutoff_index - 95, cutoff_index + 1)
        ]
        row["context_observed"] = [
            [registry["intervals"][v]["observed"] for v in slots] for slots in row["context_ids"]
        ]
        row["future_chain_ids"] = []
        broken = None
        for step in range(1, 10):
            index = cutoff_index + step
            if broken is None and config_at(index) != name:
                broken = index
            row["future_chain_ids"].append(actual(index, 0) if broken is None else None)
        row["future_absence_ref"] = None
        if broken is not None:
            ref = row["row_id"] + ":future-break"
            row["future_absence_ref"] = ref
            next_name = config_at(broken)
            reason = "PING_QUARANTINE" if next_name == "native-165" else "CONFIGURATION_BOUNDARY"
            gap = {
                "complete": True,
                **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
                "first_source_interval_index": broken,
                "last_source_interval_index": cutoff_index + 9,
                "break_source_interval_index": broken,
                "reason": reason,
                "next_configuration": next_name,
                "source_metadata_sha256": source_digest,
            }
            if reason == "PING_QUARANTINE":
                gap["quarantined_pings"] = [165]
            registry.setdefault("source_gaps", {})[ref] = gap
        row["target_requested_indices"] = [cutoff_index + h for h in (1, 3, 6)]
        (
            row["target_slot_metadata"],
            row["target_slot_processing_id_or_unknown"],
            row["target_slot_ping_counts"],
            row["target_slot_observed"],
        ) = [], [], [], []
        row["target_absence_refs"] = [None] * 3
        for j, horizon in enumerate((1, 3, 6)):
            index = cutoff_index + horizon
            identity = row["future_chain_ids"][horizon - 1]
            meta = np.zeros((4, 10), np.float32)
            if identity is not None:
                cfg = configurations[name]
                meta[:, 0] = np.asarray(cfg["frequency_hz"]) / 455000
                meta[:, 1:3] = 1
                meta[:, 3:5] = np.asarray(cfg["channel_bounds_m"]) / 250
                meta[:, 6] = 1
                meta[:, 9] = horizon
                row["target_ids"][j] = identity
                row["target_observed"][j] = True
                processing, pings, observed = (
                    cfg["processing_id_or_unknown"],
                    cfg["ping_counts"],
                    [True] * 4,
                )
            else:
                row["target_ids"][j] = index  # Actual reader preserves positive nominal requests.
                row["target_observed"][j] = False
                ref = row["future_absence_ref"]
                row["target_absence_refs"][j] = ref
                processing, pings, observed = ["UNKNOWN"] * 4, [0] * 4, [False] * 4
            row["target_slot_metadata"].append(meta.tolist())
            row["target_slot_processing_id_or_unknown"].append(list(processing))
            row["target_slot_ping_counts"].append(list(pings))
            row["target_slot_observed"].append(list(observed))
    used = {v for r in registry["rows"] for slots in r["context_ids"] for v in slots}
    used.update(v for r in registry["rows"] for v in r["future_chain_ids"] if isinstance(v, str))
    registry["intervals"] = {v: r for v, r in registry["intervals"].items() if v in used}
    return registry, source_evidence


def native_case(prefix, monkeypatch, root):
    holder = case(prefix, monkeypatch, root)
    raw, source_evidence = native_configuration_metadata(
        cutoffs=tuple(range(18)) + tuple(range(888, 906))
    )
    holder.registry = raw
    holder.prefix_arrays["x"][
        ~np.isfinite(holder.prefix_arrays["x"])
    ] = -50.0  # Synthetic fixture values only.
    holder.documents["raw.json"] = raw
    holder.documents["sources.txt"]["sources"] = raw["sources"]
    holder.documents["native-configurations.json"] = raw["configuration_map"]
    holder.documents["native-source-metadata.json"] = source_evidence
    holder.manifest["configuration_map"] = "native-configurations.json"
    holder.documents["source-gaps.json"] = {
        "kind": "native_prefix_source_gaps_v1",
        "complete": True,
        "gaps": raw.get("source_gaps", {}),
        "source_metadata": {
            ref: "native-source-metadata.json" for ref in raw.get("source_gaps", {})
        },
    }
    if raw.get("source_gaps"):
        holder.manifest["source_gaps"] = "source-gaps.json"
    for row in raw["rows"]:
        index = next(
            (i for i, r in enumerate(holder.prefix_arrays["row_id"]) if r == row["row_id"]), None
        )
        if index is not None:
            holder.prefix_arrays["target_observed"][index] = row["target_observed"]
            holder.prefix_arrays["context_observed"][index] = row["context_observed"]
            cfg = raw["configuration_map"]["sources"][row["deployment"]]["configurations"][
                row["configuration"]
            ]
            holder.prefix_arrays["metadata"][index, :, 3:5] = (
                np.asarray(cfg["channel_bounds_m"]) / 250
            )
            holder.prefix_arrays["metadata"][index, :, 6] = [
                p != "UNKNOWN" for p in cfg["processing_id_or_unknown"]
            ]
            holder.prefix_arrays["query"][index, :, :9] = holder.prefix_arrays["metadata"][
                index, :1, :9
            ]
    holder.seal()
    return holder


def direct_ancestry(updates=4, selected=1):
    return {
        "mode": "direct_end_to_end",
        "ssl_only": False,
        "supervised_updates": updates,
        "selected_supervised_step": selected,
        "ancestor_encoder_sha256": None,
        "ancestor_run_sha256": None,
    }


def frozen_case(prefix, monkeypatch, root, family="core", method="direct"):
    holder = case(prefix, monkeypatch, root, mode="frozen_readout")
    holder.config = prefix.PrefixConfig(
        **{**holder.config.to_dict(), "family": family, "method": method}
    )
    holder.manifest["config"] = holder.config.to_dict()
    holder.documents["config.json"] = holder.config.to_dict()
    holder.backbone["method"] = method
    if family == "band":
        holder.backbone["architecture"] = prefix.ARCHITECTURES["band"]
        holder.manifest["architecture"] = prefix.ARCHITECTURES["band"]
    config = holder.config
    import torch

    if family == "band":
        from marine_echo.models.native_band_temporal import NativeBandTemporalModel as ParentModel
        from marine_echo.training import native_band_ssl as original_core
        from marine_echo.training.native_band_downstream import (
            DownstreamConfig,
            RunInputs,
            required_paths,
        )
    else:
        from marine_echo.models.native_temporal import NativeTemporalModel as ParentModel
        from marine_echo.training import native_ssl as original_core
        from marine_echo.training.native_downstream import (
            DownstreamConfig,
            RunInputs,
            required_paths,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = ParentModel(8, 4, 1, 2).eval()
    downstream = DownstreamConfig(
        method="direct", mode="direct_end_to_end", updates=4, cadence=1, batch_size=8
    ).to_dict()
    holder.documents["parent-config.json"] = downstream
    holder.fs.files[holder.base / "adr0016.txt"] = b"SYNTHETIC_CORRECTNESS_ONLY original ADR0016"
    holder.fs.files[holder.base / "downstream-protocol.txt"] = (
        b"SYNTHETIC_CORRECTNESS_ONLY original ADR0018"
    )
    input_paths = {
        "train": "train.npz",
        "dev": "dev.npz",
        "train_cohort": "train-cohort.json",
        "dev_cohort": "dev-cohort.json",
        "split": "split.txt",
        "adr0016": "adr0016.txt",
        "protocol": "downstream-protocol.txt",
        "config": "parent-config.json",
        "review": "parent-review.json",
    }
    holder.documents["parent-inputs.json"] = input_paths
    inputs = RunInputs(**{k: holder.base / v for k, v in input_paths.items()})
    holder.seal()
    bindings = {
        str(p): prefix.sha(p.read_bytes())
        for p in required_paths(inputs, original_core.Config(**holder.backbone))
    }
    bindings[
        str(
            Path(original_core.__file__).with_name(
                "native_band_downstream.py" if family == "band" else "native_downstream.py"
            )
        )
    ] = prefix.sha(
        Path(original_core.__file__)
        .with_name("native_band_downstream.py" if family == "band" else "native_downstream.py")
        .read_bytes()
    )
    for name in ("train.npz", "dev.npz", "split.txt", "parent-config.json"):
        bindings[str(holder.base / name)] = prefix.sha(holder.fs.files[holder.base / name])
    parent_review = {
        "status": "APPROVED_DOWNSTREAM_PREFIT",
        "implementer_session_id": prefix.IMPLEMENTER_SESSION_ID,
        "reviewer_session_id": "synthetic-parent-reviewer",
        "allowed_methods": ["direct"],
        "allowed_modes": ["direct_end_to_end"],
        "bindings": bindings,
        "train_npz_sha256": bindings[str(holder.base / "train.npz")],
        "dev_npz_sha256": bindings[str(holder.base / "dev.npz")],
        "split_sha256": bindings[str(holder.base / "split.txt")],
    }
    holder.documents["parent-review.json"] = parent_review
    lineage = direct_ancestry()
    selected = {
        "kind": prefix.selected_kind(config),
        "config": holder.backbone,
        "scalers": holder.documents["stats.json"],
        "bindings": bindings,
        "encoder": prefix.core.cpu_state(model.encoder),
    }
    inference = {
        "kind": "native_band_ssl_weights_only_inference_v1"
        if family == "band"
        else "native_ssl_weights_only_inference_v1",
        "config": holder.backbone,
        "scalers": holder.documents["stats.json"],
        "bindings": bindings,
        "model": prefix.core.cpu_state(model),
    }
    if family == "band":
        selected["architecture"] = inference["architecture"] = prefix.ARCHITECTURES["band"]
    if method == "direct":
        selected.update(supervised_ancestry=lineage, evidence_kind=prefix.EVIDENCE)
        inference.update(
            supervised_ancestry=lineage,
            evidence_kind=prefix.EVIDENCE,
            downstream_config=downstream,
            review_sha256=prefix.sha(encoded(parent_review)),
        )
    holder.fs.files[holder.base / "selected.pt"] = prefix.encode_checkpoint(selected)
    holder.fs.files[holder.base / "parent-inference.pt"] = prefix.encode_checkpoint(inference)
    train_rows = ["SYNTHETIC_CORRECTNESS_ONLY-t0", "SYNTHETIC_CORRECTNESS_ONLY-t1"]
    sequence = []
    for i in range(4):
        ids = prefix.core.batch_indices(np.arange(2), 8, 7, "readout", i)
        sequence.append(
            {
                "step": i,
                "indices": ids.tolist(),
                "row_ids": [train_rows[j] for j in ids],
                "sha256": prefix.core.sequence_hash(ids),
            }
        )
    membership = {
        "train_row_ids": train_rows,
        "train_deployments": ["synthetic-TRAIN-deployment"] * 2,
        "train_archive_sha256": ["synthetic-TRAIN-archive"] * 2,
        "supervised_indices": [0, 1],
        "sequence": sequence,
        "sequence_sha256": prefix.sha(
            json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    holder.fs.files[holder.base / "membership.json"] = encoded(membership)
    holder.documents["parent-run.json"] = {
        "config": holder.backbone,
        "bindings": bindings,
        "inference_sha256": prefix.sha(holder.fs.files[holder.base / "parent-inference.pt"]),
        "membership_sha256": prefix.sha(holder.fs.files[holder.base / "membership.json"]),
    }
    if method == "direct":
        holder.documents["parent-run.json"].update(
            config=downstream,
            core_config=holder.backbone,
            mode="direct_end_to_end",
            status="COMPLETED",
            evidence_kind=prefix.EVIDENCE,
            test_access="NOT_RUN",
            selected_encoder_sha256=prefix.sha(holder.fs.files[holder.base / "selected.pt"]),
            review_sha256=prefix.sha(encoded(parent_review)),
            reviewer_session_id="synthetic-parent-reviewer",
            supervised_ancestry=lineage,
            supervised_updates=4,
            selected_supervised_step=1,
        )
        holder.manifest.update(
            parent_config="parent-config.json",
            parent_review="parent-review.json",
            parent_inputs="parent-inputs.json",
        )
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
    holder.parent_inference = inference
    holder.seal()
    return holder
