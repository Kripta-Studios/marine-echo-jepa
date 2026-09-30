"""Additive native held-out materialization with exact source interval provenance.

No fitting, inference, model selection, or approval is performed here. Production
decoding requires independently frozen selection and numeric-access receipts.
Source-calendar clocks are unknown UTC; centre +/-30 min is only a proxy.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
import os
import stat
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from marine_echo.data import native_ssl_corpus as reader

IMPLEMENTER = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
EVIDENCE = "REVIEWED_NATIVE_NUMERIC_MATERIALIZATION"
RAM_LIMIT = 22 * 1024**3
RESOURCE_POLICY = {
    "device": "cpu",
    "owned_tree_rss_limit_bytes": RAM_LIMIT,
    "single_owned_process": True,
}
FREQUENCIES = (38000, 125000, 200000, 455000)
MISSING = {"MISSING_CHANNEL", "DUPLICATE_ROW"}
LOADED_READER_HASH = reader.sha256(Path(reader.__file__))


def encode_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON constant.")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise TypeError("JSON object required.")
    return result


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit immutable path required.")
    value = Path(value)
    return Path(os.path.abspath(value if value.is_absolute() else base / value))


def _no_links(path):
    for value in (path, *path.parents):
        if value.exists() or value.is_symlink():
            record = value.lstat()
            if stat.S_ISLNK(record.st_mode) or getattr(record, "st_file_attributes", 0) & 0x400:
                raise ValueError("Symlink/reparse-point identities are unsupported.")


def _regular(path):
    _no_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Safe regular input file required.")
    return path


def _date(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError("Source clock timezone is unknown; no UTC conversion.")
    return result


def _review_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Approval chronology needs explicit timezone.")
    return result


def _distinct(review, manifest, status):
    excluded = {IMPLEMENTER, manifest["implementer_session_id"], manifest["coordinator_session_id"]}
    reviewer = review.get("reviewer_session_id")
    if (
        review.get("status") != status
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or reviewer.casefold() in {v.casefold() for v in excluded}
    ):
        raise ValueError("Genuine distinct approval required.")
    if review.get("implementer_session_id") != manifest["implementer_session_id"]:
        raise ValueError("Review implementer identity differs.")


def required_sources():
    """Static local import closure; no provenance-directed imports or commands."""
    package = Path(reader.__file__).resolve().parents[1]
    own = Path(__file__).resolve()
    cli = own.parents[3] / "tools/materialize_native_transfer_corpus.py"
    pending = [
        own,
        cli,
        package / "data/native_ssl_corpus.py",
        package / "training/native_prefix_transfer.py",
        package / "training/native_references.py",
    ]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if name.split(".")[0] != "marine_echo":
                    continue
                parts = name.split(".")[1:]
                for n in range(len(parts) + 1):
                    candidate = package.joinpath(*parts[:n])
                    for source in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if source.is_file():
                            pending.append(source)
    return tuple(sorted(found))


@dataclass(frozen=True)
class Admission:
    manifest: dict
    review: dict
    split: dict
    output: Path
    receipt: Path
    bindings: dict
    paths: dict
    review_sha256: str


def admit(manifest_path, review_path, output_path, receipt_path):
    """Hash/identity/freeze gate BEFORE archive parsing or output mutation."""
    manifest_path = _regular(_path(str(manifest_path), Path.cwd()))
    review_path = _regular(_path(str(review_path), Path.cwd()))
    review_raw = review_path.read_bytes()
    m, review = _json(manifest_path.read_bytes()), _json(review_raw)
    if (
        m.get("kind") != "native_transfer_materialization_manifest_v1"
        or m.get("role") != "final_test"
        or m.get("evidence_kind") != EVIDENCE
        or m.get("history") != 96
        or type(m.get("history")) is not int
        or m.get("device") != "cpu"
        or m.get("fit_authorized") is not False
    ):
        raise ValueError(
            "Explicit selection-free final-test CPU materialization contract required."
        )
    for key in ("implementer_session_id", "coordinator_session_id"):
        if not isinstance(m.get(key), str) or not m[key].strip():
            raise ValueError("Explicit session identities required.")
    if m["implementer_session_id"] != IMPLEMENTER:
        raise ValueError("Actual builder identity differs.")
    _distinct(review, m, "APPROVED_NATIVE_NUMERIC_ACCESS")
    if (
        review.get("allowed_roles") != ["final_test"]
        or review.get("evidence_kind") != EVIDENCE
        or review.get("scope") != "native_transfer_materialization"
        or review.get("fit_authorized") is not False
    ):
        raise ValueError("Exact final-test numeric scope required; no fitting admission.")
    output, receipt = (_path(str(p), manifest_path.parent) for p in (output_path, receipt_path))
    for path in (output, receipt):
        _no_links(path)
        if path.exists():
            raise FileExistsError("Fresh output directory and receipt required.")
        if not path.parent.is_dir():
            raise ValueError("Root must prepare output/receipt parents.")
    if output == receipt or output in receipt.parents or receipt in output.parents:
        raise ValueError("Output and protected receipt must be disjoint.")
    if review.get("output_path") != str(output) or review.get("receipt_path") != str(receipt):
        raise ValueError("Review runtime output identities differ.")
    if m.get("output_path") != str(output) or m.get("receipt_path") != str(receipt):
        raise ValueError("Manifest runtime output identities differ.")
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Complete nonempty exact bindings required.")
    exact = {}
    for name, digest in bindings.items():
        path = _regular(_path(name, manifest_path.parent))
        if str(path) != name or reader.sha256(path) != digest:
            raise ValueError("Stale or noncanonical numeric review binding.")
        exact[name] = digest

    def bound(path):
        path = _regular(path)
        if exact.get(str(path)) != reader.sha256(path):
            raise ValueError(f"Missing or stale required binding: {path}")
        if path == output or output in path.parents or path == receipt:
            raise ValueError("Output overlaps immutable inputs.")
        return path

    bound(manifest_path)
    for path in required_sources():
        bound(path)
    if reader.sha256(Path(reader.__file__)) != LOADED_READER_HASH:
        raise ValueError("Loaded immutable reader changed.")
    paths = {
        key: bound(_path(m[key], manifest_path.parent))
        for key in (
            "split",
            "protocol",
            "prefix_protocol",
            "dependency_lock",
            "selection",
            "selection_review",
            "output_identity",
            "supervisor",
        )
    }
    supervisor = Path(reader.__file__).resolve().parents[3] / "tools/native_reference_supervisor.py"
    prefix_protocol = (
        Path(reader.__file__).resolve().parents[3]
        / "docs/adr/0021-native-prefix-transfer-assessment.md"
    )
    if paths["prefix_protocol"] != prefix_protocol:
        raise ValueError("Exact separately versioned prefix protocol binding required.")
    if (
        paths["supervisor"] != supervisor
        or m.get("resource_policy") != RESOURCE_POLICY
        or review.get("resource_policy") != RESOURCE_POLICY
    ):
        raise ValueError("Immutable root supervisor and exact 22 GiB CPU resource policy required.")
    split = _json(paths["split"].read_bytes())
    if (
        split.get("schema_version") != "native_acoustic_ssl_v1"
        or split.get("support_history") != 96
        or "train" in split
        or "reserved_test" in split
    ):
        raise ValueError("Original native split schema required.")
    if (
        paths["protocol"]
        != (paths["split"].resolve().parents[1] / split["protocol_path"]).resolve()
    ):
        raise ValueError("Exact original split-relative protocol path required.")
    if _json(paths["output_identity"].read_bytes()) != {
        "kind": "native_transfer_output_identity_v1",
        "output_path": str(output),
        "receipt_path": str(receipt),
        "role": "final_test",
        "fit_authorized": False,
    }:
        raise ValueError("Bound output identity differs.")
    selection = _json(paths["selection"].read_bytes())
    frozen_review = _json(paths["selection_review"].read_bytes())
    _distinct(frozen_review, m, "APPROVED_SELECTION_FREEZE")
    if (
        selection.get("kind") != "native_selection_freeze_v1"
        or selection.get("selection_role") not in ("development", "prespecified_without_selection")
        or selection.get("frozen_before_numeric_access") is not True
        or selection.get("final_test_numerical_model_selection") is not False
        or selection.get("owner_session_id") != m["coordinator_session_id"]
        or selection.get("owner_freeze_confirmed") is not True
        or frozen_review.get("scope") != "selection_freeze"
        or _review_time(frozen_review["reviewed_at"]) >= _review_time(review["reviewed_at"])
        or _review_time(selection["frozen_at"]) > _review_time(frozen_review["reviewed_at"])
        or _review_time(review["reviewed_at"]) > datetime.now(UTC)
        or review.get("frozen_selection_sha256") != exact[str(paths["selection"])]
    ):
        raise ValueError("Independent selected endpoints must freeze before numeric access.")
    fb = frozen_review.get("bindings", {})
    if not fb or fb.get(str(paths["selection"])) != exact[str(paths["selection"])]:
        raise ValueError("Selection freeze itself is not independently hash-bound.")
    for name, digest in fb.items():
        if (
            exact.get(name) != digest
            or reader.sha256(_regular(_path(name, manifest_path.parent))) != digest
        ):
            raise ValueError("Stale selection-review binding.")
    artifacts = selection.get("methods")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Frozen completed model/readout/scaler/config identities required.")
    for entry in artifacts.values():
        if not isinstance(entry, dict) or entry.get("status") != "COMPLETED":
            raise ValueError("Incomplete selected ancestor cannot authorize numerical access.")
        for key in ("model", "readout", "scalers", "config", "completion", "ancestry"):
            p = bound(_path(entry[key], paths["selection"].parent))
            if fb.get(str(p)) != exact[str(p)]:
                raise ValueError("Model endpoint is not bound by both distinct reviews.")
        completed = _json(_path(entry["completion"], paths["selection"].parent).read_bytes())
        family = entry.get("artifact_family", "native_neural")
        if family == "native_neural":
            complete = (
                completed.get("status") == "COMPLETED"
                and completed.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
            )
        elif family == "native_reference":
            complete = (
                completed.get("status") == "COMPLETED_DEVELOPMENT_REFERENCE"
                and completed.get("method") in ("persistence", "seasonal24", "lightgbm")
                and completed.get("tuning") == "one_fixed_development_recipe_no_test_access"
                and completed.get("history") == 96
                and completed.get("evidence_kind") is None
            )
            for model in completed.get("models", []):
                model_path = bound(
                    _path(
                        model["path"], _path(entry["completion"], paths["selection"].parent).parent
                    )
                )
                if (
                    fb.get(str(model_path)) != exact[str(model_path)]
                    or model.get("sha256") != exact[str(model_path)]
                ):
                    raise ValueError("Every fixed booster requires both freeze/access bindings.")
            if completed.get("method") == "lightgbm" and len(completed.get("models", [])) != 15:
                complete = False
        else:
            complete = False
        if not complete:
            raise ValueError(
                "Actual completed REAL_TRAIN_DEVELOPMENT_FIT endpoint receipt required."
            )
    selected_sources, seen = [], {"deployment": set(), "archive": set(), "file_id": set()}
    declarations = m.get("source_identities", {})
    if not isinstance(split.get("sources"), list) or not split["sources"]:
        raise ValueError("Complete original source split required.")
    sites = {"train": set(), "development": set(), "final_test": set()}
    for source in split["sources"]:
        if (
            source.get("role") not in sites
            or type(source.get("file_id")) is not int
            or source["file_id"] <= 0
            or not source.get("site")
            or not source.get("deployment")
            or not isinstance(source.get("archive_sha256"), str)
            or len(source["archive_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["archive_sha256"])
        ):
            raise ValueError("Unknown/conflicting native source role or identity.")
        for key, value in (
            ("deployment", source["deployment"]),
            ("archive", source["archive_sha256"]),
            ("file_id", source["file_id"]),
        ):
            if value in seen[key]:
                raise ValueError("Duplicate source or fitted/final identity overlap.")
            seen[key].add(value)
        sites[source["role"]].add(source["site"])
        if source["role"] != "final_test":
            continue
        p = bound(_path(source["path"], paths["split"].parent))
        if exact[str(p)] != source["archive_sha256"]:
            raise ValueError("Archive and split identities differ.")
        source = copy.deepcopy(source)
        source["path"] = str(p)
        identity = declarations.get(source["deployment"], {})
        if (
            identity.get("deployment") != source["deployment"]
            or identity.get("site") != source["site"]
            or identity.get("archive") != source["archive_sha256"]
            or identity.get("native_bounds_m") != [0, 230]
            or _date(identity["start"]).strftime("%Y%m%d") != source["start_date"]
            or _date(identity["end"]).strftime("%Y%m%d")
            != (
                date(
                    int(source["end_date_inclusive"][:4]),
                    int(source["end_date_inclusive"][4:6]),
                    int(source["end_date_inclusive"][6:]),
                )
                + timedelta(days=1)
            ).strftime("%Y%m%d")
            or not isinstance(source.get("complete_ping_counts"), list)
            or not source["complete_ping_counts"]
            or len(set(source["complete_ping_counts"])) != len(source["complete_ping_counts"])
            or any(
                type(v) is not int or v not in (150, 180) for v in source["complete_ping_counts"]
            )
            or ("quarantined_ping_counts" in source and source["quarantined_ping_counts"] != [165])
        ):
            raise ValueError(
                "Exact native230 lifetime/configuration/quarantine identities required."
            )
        selected_sources.append(source)
    if sites["final_test"] & (sites["train"] | sites["development"]):
        raise ValueError("Whole final-site reservations overlap fitted/development sources.")
    if not selected_sources or set(declarations) != {s["deployment"] for s in selected_sources}:
        raise ValueError("Every final deployment must be declared; no filtering.")
    if review.get("source_identities") != declarations:
        raise ValueError("Review exact source/lifetime identities differ.")
    # The immutable gate also checks original split-relative ADR identity.
    admitted_split = reader.check_numeric_review(paths["split"], review_path, "final_test")
    if admitted_split != split or reader.sha256(paths["protocol"]) != split["protocol_sha256"]:
        raise ValueError("Original numeric/protocol admission differs.")
    if m.get("original_dev") is not None:
        dev = m["original_dev"]
        paths["dev_input"] = bound(_path(dev["input"], manifest_path.parent))
        paths["dev_cohort"] = bound(_path(dev["cohort"], manifest_path.parent))
        c = _json(paths["dev_cohort"].read_bytes())
        if (
            c.get("role") != "development"
            or c.get("input_npz_sha256") != exact[str(paths["dev_input"])]
            or c.get("split_sha256") != exact[str(paths["split"])]
            or not c.get("rows")
            or c.get("complete") is not True
            or c.get("native_bounds_m") != [0, 225]
            or review.get("original_dev_transport") != dev
        ):
            raise ValueError("Original DEV immutable identity transport differs.")
        allowed = {s["deployment"] for s in split["sources"] if s["role"] == "development"}
        if any(r[0] not in allowed for r in c["rows"]):
            raise ValueError("DEV identity transport is outside original split.")
    paths["numeric_review"] = review_path
    if reader.sha256(review_path) != _hash(review_raw):
        raise ValueError("Numeric review changed during admission.")
    return Admission(
        copy.deepcopy(m),
        copy.deepcopy(review),
        {**split, "selected_sources": selected_sources},
        output,
        receipt,
        exact,
        paths,
        _hash(review_raw),
    )


def source_clock_metadata(source):
    """Read actual Date_M/Time_M/config/QC precursor fields, never Sv_mean values."""
    records = {}
    with zipfile.ZipFile(source["path"]) as archive:
        for member in sorted(archive.namelist()):
            match = reader.MEMBER.fullmatch(member)
            if match is None:
                continue
            channel = FREQUENCIES.index(int(match.group(1)) * 1000)
            with archive.open(member) as stream:
                for ordinal, row in enumerate(
                    csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                ):
                    if not source["start_date"] <= row["Date_M"] <= source["end_date_inclusive"]:
                        continue
                    records.setdefault(int(row["Interval"]), {}).setdefault(channel, []).append(
                        {
                            "member": member,
                            "row_ordinal": ordinal,
                            "Date_M": row["Date_M"],
                            "Time_M": row["Time_M"],
                            "timestamp": str(reader._timestamp(row)),
                            "Layer": row["Layer"],
                            "upper_m": row["Layer_depth_min"],
                            "lower_m": row["Layer_depth_max"],
                            "Ping_S": row["Ping_S"],
                            "Ping_E": row["Ping_E"],
                            "Process_ID": row["Process_ID"],
                        }
                    )
    return records


def _config(slot):
    return {
        "channel_bounds_m": slot.bounds.tolist(),
        "ping_counts": list(slot.ping_counts),
        "processing_id_or_unknown": list(slot.processing),
        "frequency_hz": list(FREQUENCIES),
        "interval_seconds": 3600,
    }


def derive_metadata(slots, arrays, source, metadata_path, *, clock_records=None):
    """Derive complete raw issuance provenance without modifying reader arrays.

    NativeSlot clocks suffice only for synthetic fixtures; production passes exact
    original channel metadata so mismatched source clocks cannot be fabricated.
    """
    import numpy as np

    if not slots or len(arrays["x"]) == 0:
        raise ValueError("No native issuance; no synthetic substitution.")
    by_id = {s.interval: s for s in slots}
    if len(by_id) != len(slots) or any(s.interval < 0 for s in slots):
        raise ValueError("Unique nonnegative ordinal interval IDs required.")
    if (
        any(
            s.deployment != source["deployment"] or s.archive_sha256 != source["archive"]
            for s in slots
        )
        or any(not np.array_equal(s.bounds[0], [0, 230]) for s in slots)
        or source["native_bounds_m"] != [0, 230]
    ):
        raise ValueError("True deployment/archive/native230 identity differs.")
    if str(min(s.timestamp for s in slots)) != str(np.datetime64(source["start"], "us")):
        raise ValueError("Frozen source start must equal actual earliest primary timestamp.")
    dep = source["deployment"]
    ordered = sorted(slots, key=lambda s: s.interval)
    configs, assignment, groups = {}, {}, []
    references = []
    # Missing/partial secondary rows do not introduce an instrument configuration.
    for slot in ordered:
        if (
            groups
            and slot.interval == groups[-1][-1].interval + 1
            and all(reader._same_configuration(previous, slot) for previous in references)
        ):
            groups[-1].append(slot)
        else:
            groups.append([slot])
            references = [slot] * 4
        for channel in range(4):
            if not channel or slot.qc[channel] not in MISSING | {"PARTIAL_SOURCE_INTERVAL"}:
                references[channel] = slot
    for group in groups:
        cfg = _config(group[-1])
        for c in range(1, 4):
            candidates = [s for s in group if s.qc[c] not in MISSING | {"PARTIAL_SOURCE_INTERVAL"}]
            if candidates:
                actual = candidates[-1]
                cfg["channel_bounds_m"][c] = actual.bounds[c].tolist()
                cfg["ping_counts"][c] = actual.ping_counts[c]
                cfg["processing_id_or_unknown"][c] = actual.processing[c]
        name = "native-config:" + _hash(encode_json(cfg))
        configs[name] = cfg
        assignment.update({s.interval: name for s in group})
    first, last = ordered[0].interval, ordered[-1].interval
    segments = []
    for index in range(first, last + 1):
        name = assignment.get(index)
        reason = None
        if name is None:
            primary = [] if clock_records is None else clock_records.get(index, {}).get(0, [])
            reason = "DUPLICATE_ROW" if len(primary) > 1 else "MISSING_SOURCE_INTERVAL"
        if (
            segments
            and segments[-1]["configuration"] == name
            and segments[-1].get("reason") == reason
        ):
            segments[-1]["last_source_interval_index"] = index
        else:
            segments.append(
                {
                    "first_source_interval_index": index,
                    "last_source_interval_index": index,
                    "configuration": name,
                    **({"reason": reason} if reason else {}),
                }
            )
    evidence = {
        "kind": "native_prefix_source_configuration_metadata_v1",
        "complete": True,
        "source": copy.deepcopy(source),
        "configurations": configs,
        "segments": segments,
        "source_clock": "SOURCE_TIMEZONE_UNKNOWN",
        "interval_edges": "centre_plus_minus_30min_proxy",
        "reader_sha256": LOADED_READER_HASH,
        "channel_metadata": clock_records
        if clock_records is not None
        else {
            s.interval: {
                c: {
                    "timestamp": str(s.timestamp),
                    "bounds": s.bounds[c].tolist(),
                    "pings": s.ping_counts[c],
                    "processing": s.processing[c],
                    "qc": s.qc[c],
                }
                for c in range(4)
                if s.qc[c] not in MISSING
            }
            for s in slots
        },
        "acoustic_values": "NOT_INCLUDED",
    }
    metadata_hash = _hash(encode_json(evidence))
    entry = {
        **source,
        "configurations": configs,
        "segments": segments,
        "first_source_interval_index": first,
        "last_source_interval_index": last,
        "source_metadata_sha256": metadata_hash,
    }
    catalog = {
        "kind": "native_prefix_source_configuration_map_v1",
        "complete": True,
        "quarantined_pings": [165],
        "sources": {dep: entry},
        "source_metadata": {dep: str(metadata_path)},
    }
    registry = {
        "kind": "native_prefix_raw_intervals_v2",
        "complete": True,
        "configuration_mode": "per_source_configuration_map_v1",
        "sources": {dep: source},
        "configuration_map": catalog,
        "intervals": {},
        "rows": [],
        "source_gaps": {},
    }

    def identity(index, channel):
        slot = by_id[index]
        if slot.qc[channel] in MISSING:
            return None
        timestamp = str(slot.timestamp)
        if clock_records is not None:
            originals = clock_records.get(index, {}).get(channel, [])
            if len(originals) != 1:
                raise ValueError("Actual raw channel identity missing or duplicated.")
            timestamp = originals[0]["timestamp"]
            if timestamp != str(slot.timestamp):
                raise ValueError(
                    "Actual secondary source clock cannot be fabricated to match primary."
                )
        key = f"native-raw:{source['archive']}:{index}:{channel}"
        registry["intervals"][key] = {
            "deployment": dep,
            "site": source["site"],
            "archive": source["archive"],
            "configuration": assignment[index],
            "channel": channel,
            "source_interval_index": index,
            "timestamp": timestamp,
            "native_bounds_m": slot.bounds[channel].tolist(),
            "pings": slot.ping_counts[channel],
            "processing_id_or_unknown": slot.processing[channel],
            "observed": bool(slot.observed[channel]),
            "qc": slot.qc[channel],
        }
        return key

    def gap(row, lo, hi, reason, **extra):
        item = {
            **{k: row[k] for k in ("deployment", "site", "archive", "configuration")},
            "complete": True,
            "first_source_interval_index": lo,
            "last_source_interval_index": hi,
            "reason": reason,
            "source_metadata_sha256": metadata_hash,
            **extra,
        }
        ref = "native-gap:" + _hash(encode_json(item))
        registry["source_gaps"][ref] = item
        return ref

    for i, row_id in enumerate(arrays["row_id"]):
        cutoff = int(arrays["cutoff"][i])
        cfg = assignment[cutoff]
        row = {
            "deployment": dep,
            "site": source["site"],
            "archive": source["archive"],
            "configuration": cfg,
            "row_id": str(row_id),
            "cutoff": str(by_id[cutoff].timestamp),
            "context_ids": [],
            "context_absence_refs": [],
            "context_observed": arrays["observed"][i].tolist(),
            "target_observed": arrays["y_observed"][i].tolist(),
            "target_requested_indices": [cutoff + h for h in (1, 3, 6)],
            "target_slot_metadata": arrays["future_metadata"][i, :, 0].tolist(),
            "target_slot_observed": arrays["future_observed"][i, :, 0].tolist(),
            "target_slot_ping_counts": arrays["future_ping_counts"][i, :, 0].tolist(),
            "target_slot_processing_id_or_unknown": arrays["future_processing_id_or_unknown"][
                i, :, 0
            ].tolist(),
        }
        for index in arrays["context_ids"][i]:
            ids, refs = [], []
            for c in range(4):
                key = identity(int(index), c)
                ids.append(key)
                refs.append(
                    gap(row, int(index), int(index), by_id[int(index)].qc[c], channel=c)
                    if key is None
                    else None
                )
            row["context_ids"].append(ids)
            row["context_absence_refs"].append(refs)
        chain = []
        previous, broken, absence = by_id[cutoff], False, None
        # Reuse the immutable reader's actual future availability as authority.
        available_indices = set(
            arrays["future_ids"][i][np.any(arrays["future_metadata"][i], axis=(2, 3))].tolist()
        )
        for step in range(1, 10):
            index = cutoff + step
            current = by_id.get(index)
            present = index in available_indices
            if not broken and present:
                if current is None or assignment[index] != cfg:
                    raise ValueError(
                        "Derived native configuration disagrees with reader future support."
                    )
                chain.append(identity(index, 0))
                previous = current
                continue
            chain.append(None)
            if broken:
                continue
            broken = True
            extra = {"break_source_interval_index": index}
            if current is None:
                segment = next(
                    s
                    for s in segments
                    if s["first_source_interval_index"] <= index <= s["last_source_interval_index"]
                )
                reason = segment.get("reason", "MISSING_SOURCE_INTERVAL")
            elif current.ping_counts[0] == 165:
                reason, extra["quarantined_pings"] = "PING_QUARANTINE", [165]
            elif assignment[index] != cfg:
                reason, extra["next_configuration"] = "CONFIGURATION_BOUNDARY", assignment[index]
            elif not reader._consecutive(previous, current):
                reason = "SOURCE_GAP"
                extra.update(
                    previous_timestamp=str(previous.timestamp),
                    break_timestamp=str(current.timestamp),
                )
            else:
                raise ValueError(
                    "Reader structural boundary lacks representable exact metadata proof."
                )
            absence = gap(row, index, cutoff + 9, reason, **extra)
        row["future_chain_ids"], row["future_absence_ref"] = chain, absence
        row["target_ids"] = [
            chain[h - 1] if chain[h - 1] is not None else cutoff + h for h in (1, 3, 6)
        ]
        row["target_absence_refs"] = [
            None if isinstance(v, str) else absence for v in row["target_ids"]
        ]
        registry["rows"].append(row)
    gaps = {
        "kind": "native_prefix_source_gaps_v1",
        "complete": True,
        "gaps": registry["source_gaps"],
        "source_metadata": {ref: str(metadata_path) for ref in registry["source_gaps"]},
    }
    return {
        "registry": registry,
        "configuration_receipt": evidence,
        "configuration_map": catalog,
        "gap_receipt": gaps,
    }


def validate_partitions(registry):
    """Root-bound production closure is checked before calling this validator."""
    from marine_echo.training.native_prefix_transfer import partitions

    return {days: partitions(registry, days) for days in (1, 7, 30)}


def _check_bindings(admission):
    if reader.sha256(_regular(admission.paths["numeric_review"])) != admission.review_sha256:
        raise ValueError("Numeric approval changed before source decode.")
    for name, digest in admission.bindings.items():
        if reader.sha256(_regular(Path(name))) != digest:
            raise ValueError("Admitted source/artifact changed before materialization.")


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write(encode_json(value))
    return reader.sha256(path)


def _write_npz(path, arrays):
    import numpy as np

    # Real codec through an exclusive Python file handle, including Unicode paths.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return reader.sha256(path)


def _cohort(arrays, sources, role, evidence, digest, split_hash):
    return {
        "kind": "native_transfer_cohort_v1",
        "role": role,
        "evidence_kind": evidence,
        "complete": True,
        "native_bounds_m": [0, 230],
        "input_npz_sha256": digest,
        "split_sha256": split_hash,
        "rows": list(
            map(list, zip(arrays["deployment"].tolist(), arrays["row_id"].tolist(), strict=True))
        ),
        "identities": [
            {
                "deployment_id": s["deployment"],
                "site_id": s["site"],
                "archive_id": str(s["file_id"]),
                "source_ids": [s["archive_sha256"]],
            }
            for s in sources
        ],
    }


def array_provenance(arrays):
    """Bounded content identities; do not expand acoustic arrays in JSON."""
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "content_sha256_c_order": _hash(value.tobytes(order="C")),
        }
        for name, value in arrays.items()
    }


def write_source_products(directory, arrays, bundle, source, split_hash, *, evidence_kind):
    """Exclusive synthetic-testable transport; production calls only after admit."""
    import numpy as np

    if evidence_kind not in (EVIDENCE, "SYNTHETIC_CORRECTNESS_ONLY"):
        raise ValueError("Explicit production or private synthetic transport identity required.")
    synthetic = evidence_kind == "SYNTHETIC_CORRECTNESS_ONLY"
    if synthetic and ("SYNTHETIC" not in source["deployment"] or "SYNTHETIC" not in str(directory)):
        raise ValueError("Synthetic transport cannot relabel a public deployment/output.")
    directory.mkdir()
    outputs = {}

    def save(name, value):
        path = directory / name
        outputs[name] = _write_json(path, value)

    save("source-metadata.json", bundle["configuration_receipt"])
    save("configuration-map.json", bundle["configuration_map"])
    save("raw-intervals.json", bundle["registry"])
    save("source-gaps.json", bundle["gap_receipt"])
    save(
        "source-manifest.json",
        {"sources": bundle["registry"]["sources"], "quarantined_pings": [165]},
    )
    partitions = validate_partitions(bundle["registry"])
    full = {
        **arrays,
        "corpus_role": np.asarray("final_test"),
        "split_sha256": np.asarray(split_hash),
    }
    if synthetic:
        full.update(
            evidence_kind=np.asarray(evidence_kind), fixture_identity=np.asarray(evidence_kind)
        )
    full_hash = _write_npz(directory / "zero-shot.npz", full)
    outputs["zero-shot.npz"] = full_hash
    save(
        "zero-shot-cohort.json",
        _cohort(
            arrays,
            [source],
            "final_test",
            evidence_kind if synthetic else "REVIEWED_FROZEN_ASSESSMENT",
            full_hash,
            split_hash,
        ),
    )
    index = {
        (str(d), str(r)): i
        for i, (d, r) in enumerate(zip(arrays["deployment"], arrays["row_id"], strict=True))
    }
    for days, partition in partitions.items():
        selected = [index[tuple(pair)] for pair in partition["fit_rows"]]
        # NO suffix numerical arrays are placed in any fitting input.
        prefix = {k: v[selected] for k, v in arrays.items()}
        prefix.update(
            context_observed=prefix["observed"],
            targets=prefix["y"],
            target_observed=prefix["y_observed"],
            corpus_role=np.asarray("prefix"),
            evidence_kind=np.asarray(evidence_kind if synthetic else "REVIEWED_PREFIX_TRANSFER"),
            split_sha256=np.asarray(split_hash),
        )
        if synthetic:
            prefix["fixture_identity"] = np.asarray(evidence_kind)
        # Context-only fitting schema deliberately omits future-crop values.
        for key in tuple(prefix):
            if key.startswith("future"):
                prefix.pop(key)
        name = f"prefix-{days}d.npz"
        digest = _write_npz(directory / name, prefix)
        outputs[name] = digest
        save(
            f"prefix-{days}d-cohort.json",
            _cohort(
                prefix,
                [source],
                "prefix",
                evidence_kind if synthetic else "REVIEWED_PREFIX_TRANSFER",
                digest,
                split_hash,
            ),
        )
        save(
            f"partition-{days}d.json",
            {
                **partition,
                "prefix_npz_sha256": digest,
                "source_clock": "SOURCE_TIMEZONE_UNKNOWN",
                "suffix_numeric_path": None,
            },
        )
    save(
        "suffix-identities.json",
        {
            "kind": "native_prefix_common_suffix_v1",
            "role": "prefix_transfer_suffix",
            "rows": partitions[1]["suffix_rows"],
            "support": partitions[1]["reserved_suffix_support"],
            "support_sha256": partitions[1]["suffix_support_sha256"],
            "status": partitions[1]["suffix_support_status"],
            "numeric_path": None,
        },
    )
    return outputs


def materialize(manifest_path, review_path, output_path, receipt_path):
    """Actual root-only CLI operation; no public scientific execution by builder."""
    admitted = admit(manifest_path, review_path, output_path, receipt_path)
    import numpy as np
    import psutil

    started = time.monotonic()
    process, peak = psutil.Process(), 0

    def resources():
        nonlocal peak
        rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
        peak = max(peak, rss)
        if peak >= RAM_LIMIT:
            raise MemoryError("Owned native materialization RSS reaches 22 GiB limit.")

    _check_bindings(admitted)
    # Lock is within the new exclusively-created output; failures preserve it.
    admitted.output.mkdir()
    _write_json(
        admitted.output / "materialization-owner.json",
        {
            "pid": os.getpid(),
            "operation": "native_transfer_materialization",
            "device": "cpu",
            "fitting": False,
        },
    )
    reports, pieces, products = [], [], {}
    for source in admitted.split["selected_sources"]:
        _check_bindings(admitted)
        resources()
        slots = reader.read_source(source)
        arrays = reader.issue_windows(slots, history=96)
        clock = source_clock_metadata(source)
        identity = admitted.manifest["source_identities"][source["deployment"]]
        directory = admitted.output / ("source-" + str(source["file_id"]))
        bundle = derive_metadata(
            slots, arrays, identity, directory / "source-metadata.json", clock_records=clock
        )
        resources()
        products[source["deployment"]] = write_source_products(
            directory,
            arrays,
            bundle,
            source,
            admitted.bindings[str(admitted.paths["split"])],
            evidence_kind=admitted.manifest["evidence_kind"],
        )
        pieces.append(arrays)
        reports.append(
            {
                "deployment": source["deployment"],
                "site": source["site"],
                "archive_sha256": source["archive_sha256"],
                "slots": len(slots),
                "issued": len(arrays["x"]),
                "observed_target_counts": arrays["y_observed"].sum(axis=0).tolist(),
                "array_provenance": array_provenance(arrays),
                "native_bounds": sorted({tuple(s.bounds[0]) for s in slots}),
                "qc": {
                    q: sum(q in s.qc for s in slots)
                    for q in sorted({q for s in slots for q in s.qc})
                },
            }
        )
    joined = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    joined.update(
        corpus_role=np.asarray("final_test"),
        split_sha256=np.asarray(admitted.bindings[str(admitted.paths["split"])]),
    )
    resources()
    joined_hash = _write_npz(admitted.output / "zero-shot-all.npz", joined)
    _write_json(
        admitted.output / "zero-shot-all-cohort.json",
        _cohort(
            joined,
            admitted.split["selected_sources"],
            "final_test",
            "REVIEWED_FROZEN_ASSESSMENT",
            joined_hash,
            admitted.bindings[str(admitted.paths["split"])],
        ),
    )
    if "dev_input" in admitted.paths:
        # Transport original bytes, never normalize/refit or decode original DEV.
        with (admitted.output / "original-dev.npz").open("xb") as stream:
            stream.write(admitted.paths["dev_input"].read_bytes())
        with (admitted.output / "original-dev-cohort.json").open("xb") as stream:
            stream.write(admitted.paths["dev_cohort"].read_bytes())
    _check_bindings(admitted)
    resources()
    result = {
        "kind": "native_transfer_materialization_completion_v1",
        "status": "MATERIALIZED_REAL_NATIVE_PRODUCTS",
        "role": "final_test",
        "evidence_kind": EVIDENCE,
        "fitting": False,
        "model_selection": False,
        "pid": os.getpid(),
        "device": "cpu",
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_owned_rss_bytes": peak,
        "rss_limit_bytes": RAM_LIMIT,
        "external_supervisor_required": True,
        "issued": len(joined["x"]),
        "ssl_eligible": int(joined["ssl_eligible"].sum()),
        "source_reports": reports,
        "bindings": admitted.bindings,
        "numeric_review_sha256": reader.sha256(Path(review_path)),
        "products": products,
        "npz_sha256": joined_hash,
        "source_clock": "SOURCE_TIMEZONE_UNKNOWN",
        "interval_edges": "centre_plus_minus_30min_proxy",
        "metadata_fields": list(reader.METADATA_FIELDS),
        "censoring": "UNKNOWN_ABSOLUTE_CENSORING_SENTINEL_MISSING_PARTIAL_QC_EXPLICIT",
        "native_product": "published_integrated_products_no_profile_or_biomass_claim",
        "public_assessment_results": "NOT_RUN",
        "fit_approval": "NOT_GRANTED",
    }
    _write_json(admitted.output / "completion.json", result)
    _write_json(
        admitted.receipt,
        {
            "kind": "native_transfer_materialization_receipt_v1",
            "completion_sha256": reader.sha256(admitted.output / "completion.json"),
            "output": str(admitted.output),
            "status": result["status"],
        },
    )
    return result
