"""Private tiny safe serialized artifacts; no model, corpus codec or fitting."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import torch

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
BASE = Path(__file__).resolve().parent


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "native_inventory_fixture_candidate",
        BUILDER / "src/marine_echo/evaluation/native_ancestry_inventory.py",
    )
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    return candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def defaults(module, cls):
    source = MAIN / f"src/marine_echo/training/{module}.py"
    node = next(
        n
        for n in ast.parse(source.read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == cls
    )
    result = {}
    for field in node.body:
        if not isinstance(field, ast.AnnAssign):
            continue
        if field.target.id == "cf_source":
            result[field.target.id] = "Z:/INACCESSIBLE_SOURCE_METADATA_ONLY/cf"
        elif field.target.id == "architecture":
            result[field.target.id] = "nonlinear_frequency_conditioned_v1"
        else:
            result[field.target.id] = ast.literal_eval(field.value)
    return result


class Fixture:
    def __init__(self):
        self.candidate = load_candidate()
        self.root = BASE / ("SYNTHETIC_CORRECTNESS_ONLY-Árbol-" + str(uuid.uuid4()))
        self.root.mkdir()
        self.train = self.root / "train.identity-bytes"
        self.train.write_bytes(b"SYNTHETIC_CORRECTNESS_ONLY: no NPZ, no acoustic values")
        self.dev = self.root / "dev.identity-bytes"
        self.dev.write_bytes(b"SYNTHETIC_CORRECTNESS_ONLY: no development numbers")
        self.split = self.root / "split.json"
        sources = [
            {
                "role": role,
                "file_id": file_id,
                "deployment": deployment,
                "site": site,
                "archive_sha256": archive,
            }
            for role, file_id, deployment, site, archive in (
                ("train", 18593, "SYNTHETIC_TRAIN", "SYNTHETIC_AEON4_TRAIN", "a" * 64),
                ("development", 18594, "SYNTHETIC_DEV", "SYNTHETIC_DEV_SITE", "b" * 64),
                ("final_test", 18595, "SYNTHETIC_TEST", "SYNTHETIC_AEON2_RESERVED", "c" * 64),
            )
        ]
        write(
            self.split,
            {"schema_version": "native_acoustic_ssl_v1", "support_history": 96, "sources": sources},
        )
        self.rows = [["SYNTHETIC_TRAIN", f"synthetic-row-{i}"] for i in range(3)]
        self.report = self.root / "train-report.json"
        write(
            self.report,
            {
                "status": "SYNTHETIC_CORRECTNESS_ONLY",
                "role": "train",
                "issued": 3,
                "ssl_eligible": 3,
                "npz_sha256": digest(self.train),
                "split_sha256": digest(self.split),
                "source_reports": [{"deployment": "SYNTHETIC_TRAIN", "issued": 3}],
            },
        )
        self.cohort = self.root / "train-cohort.json"
        write(
            self.cohort,
            {
                "role": "train",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "complete": True,
                "input_npz_sha256": digest(self.train),
                "split_sha256": digest(self.split),
                "rows": self.rows,
                "identities": self.candidate._identities(sources[:1]),
            },
        )
        self.entries, self.data = {}, {}
        self.output = self.root / "Inventario-230m"
        self.manifest = self.root / "manifest.json"
        self.stats = {
            "channel_mean": [-60.0] * 4,
            "channel_std": [2.0] * 4,
            "target_mean": [-59.0] * 3,
            "target_std": [3.0] * 3,
        }

    def add(
        self,
        name,
        *,
        kind="native_ssl_weights_only_inference_v1",
        seed=7,
        method="shared_ssl",
        mode="core_frozen_readout",
        parent=None,
    ):
        directory = self.root / name
        directory.mkdir()
        ssl_module, downstream_module = self.candidate.KINDS[kind][2:]
        config = defaults(ssl_module, "Config")
        config.update(seed=seed, method=method)
        if method == "random_frozen":
            config["pretrain_updates"] = 0
        if method == "direct":
            config["pretrain_updates"] = 3000
        downstream = mode != "core_frozen_readout"
        actual_config = defaults(downstream_module, "DownstreamConfig") if downstream else config
        if downstream:
            actual_config.update(
                mode=mode,
                method=method,
                seed=seed,
                updates=2000 if mode == "frozen_readout" else 3000,
                cadence=500 if mode == "frozen_readout" else 750,
            )
        config_path, review_path = directory / "config.json", directory / "review.json"
        write(config_path, actual_config)
        bindings = {str(p): digest(p) for p in (self.train, self.dev, self.split, config_path)}
        source = (
            MAIN / f"src/marine_echo/training/{downstream_module if downstream else ssl_module}.py"
        )
        bindings[str(source)] = digest(source)
        lineage = None
        if downstream:
            lineage = {
                "mode": mode,
                "ssl_only": False,
                "supervised_updates": actual_config["updates"],
                "selected_supervised_step": actual_config["cadence"],
                "ancestor_run_sha256": digest(self.root / parent / "run.json") if parent else None,
                "ancestor_encoder_sha256": digest(self.root / parent / "selected_encoder.pt")
                if parent
                else None,
            }
            if parent:
                for filename in ("run.json", "selected_encoder.pt"):
                    path = self.root / parent / filename
                    bindings[str(path)] = digest(path)
        review = {
            "status": "APPROVED_DOWNSTREAM_PREFIT" if downstream else "APPROVED_PREFIT",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "reviewer_session_id": "SYNTHETIC_DISTINCT_REVIEWER",
            "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
            "bindings": bindings,
        }
        write(review_path, review)
        encoder = {
            "projection.weight": torch.full((2, 4), 1.0, dtype=torch.float32),
            "buffer": torch.tensor([0], dtype=torch.int64),
        }
        if parent:
            encoder = copy.deepcopy(self.data[parent]["encoder"])
        if mode == "full_finetune":
            encoder["projection.weight"] += 0.5
        head = {
            "readout.weight": torch.full((3, 2), float(len(self.entries) + 1), dtype=torch.float32)
        }
        artifact = {
            "kind": kind,
            "model": {**{"encoder." + k: v for k, v in encoder.items()}, **head},
            "config": config,
            "scalers": self.stats,
            "bindings": bindings,
        }
        selected = {
            "kind": self.candidate.KINDS[kind][1 if downstream else 0],
            "encoder": encoder,
            "config": config,
            "scalers": self.stats,
            "bindings": bindings,
        }
        if kind != "native_ssl_weights_only_inference_v1":
            for payload in (artifact, selected):
                payload.update(
                    architecture=self.candidate.ARCHITECTURE,
                    evidence_kind="SYNTHETIC_CORRECTNESS_ONLY",
                    correctness_smoke=True,
                )
        if downstream:
            artifact.update(
                downstream_config=actual_config,
                supervised_ancestry=lineage,
                evidence_kind="SYNTHETIC_CORRECTNESS_ONLY",
                review_sha256=digest(review_path),
            )
            selected.update(supervised_ancestry=lineage, evidence_kind="SYNTHETIC_CORRECTNESS_ONLY")
        torch.save(artifact, directory / "inference.pt")
        if mode == "frozen_readout":
            (directory / "selected_encoder.pt").write_bytes(
                (self.root / parent / "selected_encoder.pt").read_bytes()
            )
        else:
            torch.save(selected, directory / "selected_encoder.pt")
        write(directory / "scalers.json", self.stats)
        membership = {
            "train_deployments": [r[0] for r in self.rows],
            "train_row_ids": [r[1] for r in self.rows],
            "train_archive_sha256": ["a" * 64] * 3,
            "supervised_indices": [0, 1, 2],
            "sequence": [
                {"step": 0, "indices": [1, 0], "row_ids": [self.rows[1][1], self.rows[0][1]]}
            ],
        }
        if not downstream:
            membership["ssl_eligible_indices"] = [0, 1, 2]
        write(directory / "membership.json", membership)
        run = {
            "status": "COMPLETED",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "config": actual_config,
            "bindings": bindings,
            "test_access": "NOT_RUN",
            "pretrain_steps": 0 if method == "random_frozen" else 6000,
            "inference_sha256": digest(directory / "inference.pt"),
            "membership_sha256": digest(directory / "membership.json"),
            "review_sha256": digest(review_path),
        }
        if downstream:
            run.update(
                mode=mode,
                core_config=config,
                supervised_ancestry=lineage,
                supervised_updates=lineage["supervised_updates"],
                selected_supervised_step=lineage["selected_supervised_step"],
                selected_encoder_sha256=digest(directory / "selected_encoder.pt"),
            )
        if kind != "native_ssl_weights_only_inference_v1":
            run["architecture"] = self.candidate.ARCHITECTURE
            run["correctness_smoke"] = True
        write(directory / "run.json", run)
        self.entries[name] = {
            "directory": str(directory),
            "kind": kind,
            "method": method,
            "seed": seed,
            "mode": mode,
            "parent": parent,
            "config_path": str(config_path),
            "review_path": str(review_path),
        }
        self.data[name] = {"encoder": encoder, "artifact": artifact, "run": run}
        return self

    def finish(self, references=None):
        bindings = {str(p): digest(p) for p in self.candidate.required_sources()}
        for file in self.root.rglob("*"):
            if file.is_file() and file != self.manifest:
                bindings[str(file)] = digest(file)
        document = {
            "kind": "native_research_inventory_manifest_v1",
            "purpose": "LOCAL_METADATA_DERIVATION_ONLY",
            "execution": "ROOT_LOCAL_COMPLETED_METADATA_AUDIT",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "device": "cpu",
            "output_path": str(self.output),
            "bindings": bindings,
            "split_path": str(self.split),
            "train": {
                "input_path": str(self.train),
                "report_path": str(self.report),
                "identity_cohort_path": str(self.cohort),
            },
            "endpoints": self.entries,
            "references": references or {},
            "owner_session_id": "SYNTHETIC_ROOT",
        }
        write(self.manifest, document)
        return self

    def rewrite(self, path, document, *, rebind=False):
        # Only new private synthetic fixture bytes, never an original denied path.
        path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        if rebind and self.manifest.exists():
            manifest = json.loads(self.manifest.read_bytes())
            manifest["bindings"][str(path)] = digest(path)
            self.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
