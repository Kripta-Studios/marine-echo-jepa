"""Hash-gated prefix adaptation, separately versioned from zero-shot assessment.

ADR0021 is a proposed protocol, not permission. Root supplies complete source-clock
interval metadata and genuine distinct numeric/prefit reviews. Fitting receives
a prefix-only NPZ; no suffix numerical archive is accepted or decoded. All local
ancestry must be explicit. Source-clock timestamps are not claimed verified UTC.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marine_echo.evaluation.native_product import native_scores
from marine_echo.inference.native_encoder import TRAINING_KINDS, _inputs, _scalers
from marine_echo.models.native_band_temporal import NativeBandEncoder
from marine_echo.models.native_temporal import CFTemporalEncoder, QueryHead, SharedTemporalEncoder
from marine_echo.training import native_ssl as core

IMPLEMENTER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
EVIDENCE = "SYNTHETIC_CORRECTNESS_ONLY"
KINDS = {
    "core": "native_ssl_selected_encoder_v1",
    "band": "native_band_ssl_selected_encoder_v1",
}
REPLICATION_KIND = "native_band_replication_ssl_selected_encoder_v2"
REPLICATION_SUPERVISED_KIND = "native_band_replication_downstream_supervised_encoder_v2"
SUPERVISED_KINDS = {
    "core": "native_downstream_supervised_encoder_v1",
    "band": "native_band_downstream_supervised_encoder_v1",
}
ARCHITECTURES = {
    "core": "shared_temporal_v1",
    "band": "nonlinear_frequency_conditioned_v1",
    "cf": "cf_jepa_multiscale_v1",
}
HORIZONS = (1, 3, 6)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError("Duplicate JSON key/identity.")
        result[k] = v
    return result


def _json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON policy constant.")

    result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise TypeError("Explicit JSON object required.")
    return result


def _timestamp(value):
    if not isinstance(value, str):
        raise TypeError("Explicit source-centred timestamp required.")
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError("Explicit source-clock timestamps; no invented UTC.")
    return result


def boundaries(source_start, prefix_days):
    if type(prefix_days) is not int or prefix_days not in (1, 7, 30):
        raise ValueError("Only prespecified 1/7/30-day prefixes.")
    start = _timestamp(source_start)
    if start.hour or start.minute or start.second or start.microsecond:
        raise ValueError("Frozen source start date must be midnight.")
    label = start + timedelta(days=4)
    return label, label + timedelta(days=prefix_days), label + timedelta(days=37)


@dataclass(frozen=True)
class PrefixConfig:
    method: str = "shared_ssl"
    family: str = "core"
    band_artifact_version: int = 1
    mode: str = "frozen_readout"
    seed: int = 7
    prefix_days: int = 1
    history: int = 96
    updates: int = 2000
    cadence: int = 500
    batch_size: int = 64
    lr: float = 0.0003
    weight_decay: float = 0.0001
    clip: float = 1.0
    floor: int = 18
    device: str = "cpu"
    correctness_smoke: bool = False

    def validate(self, evidence_kind):
        if (
            any(
                type(getattr(self, k)) is not int
                for k in (
                    "seed",
                    "prefix_days",
                    "history",
                    "updates",
                    "cadence",
                    "batch_size",
                    "floor",
                )
            )
            or type(self.correctness_smoke) is not bool
        ):
            raise ValueError("Exact integer budgets and Boolean smoke guard required.")
        if self.method not in TRAINING_KINDS or self.family not in ARCHITECTURES:
            raise ValueError("Unknown native method/family.")
        if (self.method == "cf_jepa") != (self.family == "cf"):
            raise ValueError("CF family/method identity differs.")
        if self.mode not in ("frozen_readout", "scratch_direct"):
            raise ValueError("No SSL full fine-tuning or future-crop adaptation.")
        if self.mode == "scratch_direct" and (self.method != "direct" or self.family == "cf"):
            raise ValueError("Scratch is a fresh directly supervised shared backbone.")
        if self.seed not in (7, 13, 23) or self.history != 96 or self.prefix_days not in (1, 7, 30):
            raise ValueError("Frozen seed/history/prefix recipe differs.")
        if (
            type(self.band_artifact_version) is not int
            or self.band_artifact_version not in (1, 2)
            or (self.family != "band" and self.band_artifact_version != 1)
        ):
            raise ValueError("Explicit supported band artifact version required.")
        if self.family == "band" and self.band_artifact_version == 1 and self.seed != 7:
            raise ValueError(
                "Band seed7 gate requires a separately reviewed replication extension."
            )
        if (self.lr, self.weight_decay, self.clip, self.floor) != (0.0003, 0.0001, 1.0, 18):
            raise ValueError("Frozen optimizer and daily floor differ.")
        if self.device not in ("cpu", "cuda:0"):
            raise ValueError("One declared local device only.")
        if self.correctness_smoke:
            if evidence_kind != EVIDENCE or self.device != "cpu":
                raise ValueError("Small trajectories require explicit synthetic CPU evidence.")
            if not 1 <= self.updates <= 2000 or self.cadence < 1 or self.batch_size < 1:
                raise ValueError("Invalid synthetic trajectory.")
        elif (self.updates, self.cadence, self.batch_size) != (2000, 500, 64):
            raise ValueError("Real recipe is 2000/500/64 with exactly four DEV choices.")

    def to_dict(self):
        return asdict(self)


def _configuration_map(metadata):
    """Versioned metadata authority; no numerical arrays or provenance execution."""
    if metadata.get("kind") == "native_prefix_raw_intervals_v1":
        if (
            metadata.get("configuration_mode") != "legacy_single_configuration_v1"
            or metadata.get("evidence_kind") != EVIDENCE
        ):
            raise ValueError(
                "Legacy single-configuration fixtures require explicit synthetic compatibility."
            )
        return None
    catalog = metadata.get("configuration_map", {})
    sources = metadata["sources"]
    if (
        metadata.get("kind") != "native_prefix_raw_intervals_v2"
        or metadata.get("configuration_mode") != "per_source_configuration_map_v1"
        or catalog.get("kind") != "native_prefix_source_configuration_map_v1"
        or catalog.get("complete") is not True
        or catalog.get("quarantined_pings") != [165]
        or set(catalog.get("sources", {})) != set(sources)
        or set(catalog.get("source_metadata", {})) != set(sources)
    ):
        raise ValueError("Complete versioned native configuration map required.")
    for dep, source in sources.items():
        entry = catalog["sources"][dep]
        if (
            any(
                entry.get(k) != source.get(k)
                for k in ("deployment", "site", "archive", "start", "end", "native_bounds_m")
            )
            or "configuration" in source
            or "channel_bounds_m" in source
            or not isinstance(entry.get("source_metadata_sha256"), str)
            or len(entry["source_metadata_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in entry["source_metadata_sha256"])
            or not isinstance(catalog["source_metadata"][dep], str)
            or not catalog["source_metadata"][dep]
        ):
            raise ValueError("Native configuration source identity/lifetime/provenance differs.")
        configs = entry.get("configurations", {})
        if not configs:
            raise ValueError("Empty native configuration catalog.")
        for name, cfg in configs.items():
            bounds = cfg.get("channel_bounds_m", [])
            pings, processing = cfg.get("ping_counts", []), cfg.get("processing_id_or_unknown", [])
            if (
                not isinstance(name, str)
                or not name
                or len(bounds) != 4
                or any(
                    not isinstance(b, list)
                    or len(b) != 2
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in b)
                    or not 0 <= b[0] < b[1]
                    for b in bounds
                )
                or bounds[0] != [0, 230]
                or len(pings) != 4
                or any(type(v) is not int or v not in (0, 150, 165, 180) for v in pings)
                or len(processing) != 4
                or any(not isinstance(v, str) or not v.strip() for v in processing)
                or cfg.get("frequency_hz") != [38000, 125000, 200000, 455000]
                or cfg.get("interval_seconds") != 3600
            ):
                raise ValueError("Native configuration bounds/processing/pings/frequencies differ.")
        first, last = (
            entry.get("first_source_interval_index"),
            entry.get("last_source_interval_index"),
        )
        segments = entry.get("segments", [])
        if (
            type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last
            or not segments
        ):
            raise ValueError("Exact source configuration index extent required.")
        expected = first
        for segment in segments:
            lo, hi, name = (
                segment.get(k)
                for k in (
                    "first_source_interval_index",
                    "last_source_interval_index",
                    "configuration",
                )
            )
            if (
                type(lo) is not int
                or type(hi) is not int
                or lo != expected
                or not lo <= hi <= last
                or (
                    name not in configs
                    if name is not None
                    else segment.get("reason")
                    not in {
                        "SOURCE_GAP",
                        "MISSING_SOURCE_INTERVAL",
                        "DUPLICATE_ROW",
                        "MISSING_CHANNEL",
                    }
                )
            ):
                raise ValueError(
                    "Native configuration segments must preserve exact indices and gaps."
                )
            expected = hi + 1
        if expected != last + 1:
            raise ValueError("Incomplete native configuration index extent.")
    return catalog


def _configuration_at(catalog, deployment, index):
    for segment in catalog["sources"][deployment]["segments"]:
        if segment["first_source_interval_index"] <= index <= segment["last_source_interval_index"]:
            return segment["configuration"]
    raise ValueError("Requested source index lies outside configuration provenance.")


def _native_gap(metadata, catalog, row, ref, index, *, channel=0, break_index=None):
    gap = metadata.get("source_gaps", {}).get(ref, {}) if isinstance(ref, str) else {}
    lo, hi = gap.get("first_source_interval_index"), gap.get("last_source_interval_index")
    if (
        gap.get("complete") is not True
        or any(gap.get(k) != row[k] for k in ("deployment", "site", "archive", "configuration"))
        or type(lo) is not int
        or type(hi) is not int
        or not 0 <= lo <= index <= hi
        or gap.get("source_metadata_sha256")
        != catalog["sources"][row["deployment"]]["source_metadata_sha256"]
    ):
        raise ValueError("Actual structural absence requires exact bound source-gap provenance.")
    if channel:
        if gap.get("channel") != channel or gap.get("reason") not in {
            "MISSING_CHANNEL",
            "DUPLICATE_ROW",
        }:
            raise ValueError("Missing secondary raw identity requires explicit channel absence.")
        return
    if (
        type(break_index) is not int
        or gap.get("break_source_interval_index") != break_index
        or not lo <= break_index <= index
    ):
        raise ValueError("Exact first unavailable future source index required.")
    name = _configuration_at(catalog, row["deployment"], break_index)
    if gap.get("reason") == "PING_QUARANTINE":
        if (
            name is None
            or catalog["sources"][row["deployment"]]["configurations"][name]["ping_counts"][0]
            != 165
            or gap.get("quarantined_pings") != [165]
        ):
            raise ValueError("Actual 165-ping boundary remains quarantined.")
    elif gap.get("reason") == "CONFIGURATION_BOUNDARY":
        if name is None or name == row["configuration"] or gap.get("next_configuration") != name:
            raise ValueError("Actual future configuration transition proof differs.")
    elif gap.get("reason") in {
        "SOURCE_GAP",
        "MISSING_SOURCE_INTERVAL",
        "DUPLICATE_ROW",
        "MISSING_CHANNEL",
    }:
        if name is not None:
            previous, current = gap.get("previous_timestamp"), gap.get("break_timestamp")
            if (
                gap["reason"] != "SOURCE_GAP"
                or previous is None
                or current is None
                or timedelta(minutes=55)
                <= _timestamp(current) - _timestamp(previous)
                <= timedelta(minutes=65)
            ):
                raise ValueError("Unavailable source interval needs actual gap/clock metadata.")
            # Known observed identities cannot be hidden by a fabricated clock
            # break. Receipt timestamps must agree with all existing metadata.
            by_index = {
                record["source_interval_index"]: record
                for record in metadata["intervals"].values()
                if record["deployment"] == row["deployment"] and record["channel"] == 0
            }
            before, after = by_index.get(break_index - 1), by_index.get(break_index)
            if (before is not None and _timestamp(before["timestamp"]) != _timestamp(previous)) or (
                after is not None and _timestamp(after["timestamp"]) != _timestamp(current)
            ):
                raise ValueError(
                    "Structural clock absence contradicts actual raw timestamp metadata."
                )
    else:
        raise ValueError("Unknown native structural absence reason.")


def _target_view(row, cfg):
    meta = np.asarray(row.get("target_slot_metadata"), dtype=np.float64)
    masks = np.asarray(row.get("target_slot_observed"))
    pings, processes = (
        row.get("target_slot_ping_counts"),
        row.get("target_slot_processing_id_or_unknown"),
    )
    if (
        meta.shape != (3, 4, 10)
        or not np.isfinite(meta).all()
        or masks.shape != (3, 4)
        or masks.dtype != np.bool_
        or not isinstance(pings, list)
        or len(pings) != 3
        or any(len(v) != 4 or any(type(p) is not int for p in v) for v in pings)
        or not isinstance(processes, list)
        or len(processes) != 3
        or any(len(v) != 4 or any(not isinstance(p, str) or not p for p in v) for v in processes)
        or masks[:, 0].tolist() != row["target_observed"]
    ):
        raise ValueError("Exact native future slot metadata/pings/processing/masks required.")
    available = []
    for j, horizon in enumerate(HORIZONS):
        present = bool(np.any(meta[j]))
        if not present:
            if masks[j].any() or pings[j] != [0] * 4 or processes[j] != ["UNKNOWN"] * 4:
                raise ValueError("Unavailable nominal future slot has observed metadata or labels.")
        else:
            if (
                not np.allclose(
                    meta[j, :, 0], np.asarray(cfg["frequency_hz"]) / 455000, atol=1e-7, rtol=0
                )
                or not np.allclose(meta[j, :, 1:3], 1, atol=0, rtol=0)
                or not np.allclose(meta[j, :, 9], horizon, atol=0, rtol=0)
            ):
                raise ValueError("Future native frequency/interval/horizon metadata differs.")
            for c in range(4):
                if (c == 0 or masks[j, c]) and (
                    pings[j][c] != cfg["ping_counts"][c]
                    or processes[j][c] != cfg["processing_id_or_unknown"][c]
                    or not np.allclose(
                        meta[j, c, 3:5] * 250, cfg["channel_bounds_m"][c], atol=1e-5, rtol=0
                    )
                    or meta[j, c, 6] != (processes[j][c] != "UNKNOWN")
                ):
                    raise ValueError("Available native future configuration differs from issuance.")
        available.append(present)
    return available


def partitions(metadata, days):
    """Metadata-only reservation, independent of values, target masks and holes.

    Registry entries identify actual source intervals, including missing values.
    Available observations have actual raw IDs; nominal requested future indices
    remain distinct. Structural absence may retain a positive request or -1,
    with unavailable native metadata/masks and a separately admitted source-gap
    explanation. No missing timestamp or observation ID is invented.
    """
    if metadata.get("complete") is not True:
        raise ValueError("Complete root-bound raw interval provenance required.")
    sources, intervals, rows = metadata["sources"], metadata["intervals"], metadata["rows"]
    if not sources or not intervals or not rows:
        raise ValueError("Empty interval/source/issuance provenance.")
    catalog = _configuration_map(metadata)
    starts = {}
    for dep, source in sources.items():
        if (
            source["deployment"] != dep
            or source["native_bounds_m"] != [0, 230]
            or (
                catalog is None
                and (
                    source.get("channel_bounds_m", [None])[0] != [0, 230]
                    or len(source.get("channel_bounds_m", [])) != 4
                )
            )
        ):
            raise ValueError("Conflicting deployment or native230 geometry.")
        if not all(
            isinstance(source.get(k), str) and source[k]
            for k in (
                ("site", "archive", "configuration") if catalog is None else ("site", "archive")
            )
        ):
            raise ValueError("Stable source/site/archive/configuration required.")
        starts[dep] = (
            _timestamp(source["start"]),
            _timestamp(source["end"]),
            *boundaries(source["start"], days),
        )
        if starts[dep][1] <= starts[dep][-1]:
            raise ValueError("Frozen source has no declared suffix period.")
    identity_to_stamp, stamp_to_identity = {}, {}
    identity_to_index, index_to_stamp = {}, {}
    for identity, record in intervals.items():
        dep = record["deployment"]
        if dep not in sources or not identity:
            raise ValueError("Unknown raw source interval.")
        source = sources[dep]
        for k in ("site", "archive", "configuration") if catalog is None else ("site", "archive"):
            if record.get(k) != source[k]:
                raise ValueError("Raw interval crosses deployment/site/configuration.")
        if catalog is None and record.get("pings") not in (150, 180):
            raise ValueError("165-ping and unknown eligibility remain quarantined.")
        c = record["channel"]
        if type(c) is not int or c not in range(4):
            raise ValueError("Raw interval channel invalid.")
        if catalog is None and record.get("native_bounds_m") != source["channel_bounds_m"][c]:
            raise ValueError("Raw interval native geometry mismatch.")
        stamp = (dep, _timestamp(record["timestamp"]), c)
        interval_index = record.get("source_interval_index")
        if type(interval_index) is not int or interval_index < 0:
            raise ValueError("Exact nonnegative native source interval index required.")
        if catalog is not None:
            name = _configuration_at(catalog, dep, interval_index)
            if (
                name is None
                or record.get("configuration") != name
                or type(record.get("observed")) is not bool
            ):
                raise ValueError(
                    "Actual source interval configuration/observation identity differs."
                )
            cfg = catalog["sources"][dep]["configurations"][name]
            if record["observed"]:
                if (
                    record.get("pings") not in (150, 180)
                    or record["pings"] != cfg["ping_counts"][c]
                    or record.get("processing_id_or_unknown") != cfg["processing_id_or_unknown"][c]
                    or record.get("native_bounds_m") != cfg["channel_bounds_m"][c]
                ):
                    raise ValueError("Observed source interval native configuration differs.")
            elif record.get("qc") not in {
                "MISSING_CHANNEL",
                "PARTIAL_SOURCE_INTERVAL",
                "DUPLICATE_ROW",
                "INVALID_OR_SENTINEL",
                "CLOCK_MISMATCH",
                "INVALID_NATIVE_GEOMETRY",
            }:
                raise ValueError("Unobserved source interval lacks actual QC provenance.")
        index_key = (dep, interval_index, c)
        if index_key in index_to_stamp and index_to_stamp[index_key] != stamp:
            raise ValueError("Native source interval index aliases different timestamps.")
        identity_to_index[identity], index_to_stamp[index_key] = interval_index, stamp
        if not starts[dep][0] <= stamp[1] < starts[dep][1]:
            raise ValueError("Raw interval outside frozen source lifetime.")
        if stamp in stamp_to_identity and stamp_to_identity[stamp] != identity:
            raise ValueError("Different IDs alias the same actual raw interval timestamp.")
        identity_to_stamp[identity], stamp_to_identity[stamp] = stamp, identity
    fit, suffix, seen, all_used = [], [], set(), set()
    gaps = metadata.get("source_gaps", {})
    if not isinstance(gaps, dict):
        raise TypeError("Explicit source-gap mapping required.")
    gap_refs, uncertain_prefix = set(), []
    for row in rows:
        key = (row["deployment"], row["row_id"])
        if key in seen:
            raise ValueError("Duplicate issuance identity.")
        seen.add(key)
        dep = row["deployment"]
        if dep not in sources:
            raise ValueError("Unknown issuance deployment.")
        for k in ("site", "archive", "configuration") if catalog is None else ("site", "archive"):
            if row.get(k) != sources[dep][k]:
                raise ValueError("Issuance configuration/source conflict.")
        cutoff = _timestamp(row["cutoff"])
        context, targets = row["context_ids"], row["target_ids"]
        if len(row.get("target_observed", [])) != 3 or any(
            type(v) is not bool for v in row["target_observed"]
        ):
            raise ValueError("Root-bound Boolean three-horizon label support required.")
        if len(context) != 96 or any(len(v) != 4 for v in context) or len(targets) != 3:
            raise ValueError("Complete 96x4 context and three raw target IDs required.")
        last = identity_to_stamp.get(context[-1][0])
        if last != (dep, cutoff, 0):
            raise ValueError("Cutoff must equal its actual last native source timestamp.")
        cutoff_index = identity_to_index[context[-1][0]]
        cfg, context_mask = None, None
        if catalog is not None:
            name = _configuration_at(catalog, dep, cutoff_index)
            if name != row.get("configuration") or name is None:
                raise ValueError("Issued native configuration differs from exact source index.")
            cfg = catalog["sources"][dep]["configurations"][name]
            context_mask = np.asarray(row.get("context_observed"))
            if (
                context_mask.shape != (96, 4)
                or context_mask.dtype != np.bool_
                or not context_mask[:, 0].all()
                or cfg["ping_counts"][0] not in (150, 180)
            ):
                raise ValueError(
                    "Primary issuance must be complete eligible native150/180 context."
                )
        previous = None
        for t, slots in enumerate(context):
            primary = identity_to_stamp.get(slots[0])
            if primary is None or primary[0] != dep:
                raise ValueError("Missing native context identity.")
            if previous is not None and not timedelta(minutes=55) <= primary[
                1
            ] - previous <= timedelta(minutes=65):
                raise ValueError("Native context adjacency must be 55–65 source minutes.")
            previous = primary[1]
            for c, identity in enumerate(slots):
                if catalog is not None and identity is None and c and not context_mask[t, c]:
                    ref = row.get("context_absence_refs", [[None] * 4 for _ in range(96)])[t][c]
                    _native_gap(metadata, catalog, row, ref, cutoff_index + t - 95, channel=c)
                    gap_refs.add(ref)
                    continue
                if (
                    identity_to_stamp.get(identity) != (dep, primary[1], c)
                    or identity_to_index.get(identity) != cutoff_index + t - 95
                ):
                    raise ValueError("Context discontinuity, alias or raw-ID overlap.")
                if catalog is not None:
                    record = intervals[identity]
                    if record["observed"] != bool(context_mask[t, c]) or (
                        record["observed"] and record["configuration"] != row["configuration"]
                    ):
                        raise ValueError(
                            "Observed native context must match issued configuration/mask."
                        )
        available, future_chain, break_index = None, [], None
        if catalog is not None:
            available = _target_view(row, cfg)
            requested = row.get("target_requested_indices")
            if requested != [cutoff_index + h for h in HORIZONS] or any(
                type(v) is not int for v in requested
            ):
                raise ValueError("Native nominal future requests must retain exact source offsets.")
            future_chain = row.get("future_chain_ids", [])
            if len(future_chain) != 9:
                raise ValueError("Actual contiguous future metadata identities required.")
            previous = cutoff
            for step, identity in enumerate(future_chain, 1):
                if identity is None:
                    break_index = cutoff_index + step if break_index is None else break_index
                    continue
                if break_index is not None or not isinstance(identity, str):
                    raise ValueError(
                        "Contiguous native future cannot resume after structural absence."
                    )
                record = intervals.get(identity, {})
                target = identity_to_stamp.get(identity)
                if (
                    target is None
                    or target[0] != dep
                    or target[2] != 0
                    or identity_to_index.get(identity) != cutoff_index + step
                    or record.get("configuration") != row["configuration"]
                    or record.get("pings") != cfg["ping_counts"][0]
                    or record.get("processing_id_or_unknown") != cfg["processing_id_or_unknown"][0]
                    or record.get("native_bounds_m") != cfg["channel_bounds_m"][0]
                    or not timedelta(minutes=55) <= target[1] - previous <= timedelta(minutes=65)
                ):
                    raise ValueError("Actual future index/clock/configuration continuity differs.")
                previous = target[1]
            if break_index is not None:
                ref = row.get("future_absence_ref")
                _native_gap(metadata, catalog, row, ref, cutoff_index + 9, break_index=break_index)
                gap_refs.add(ref)
            elif row.get("future_absence_ref") is not None:
                raise ValueError("Complete future chain cannot claim structural absence.")
        absence_refs = row.get("target_absence_refs", [None] * 3)
        if not isinstance(absence_refs, list) or len(absence_refs) != 3:
            raise ValueError("Exact three-horizon structural absence references required.")
        for j, (h, identity) in enumerate(zip(HORIZONS, targets, strict=True)):
            if catalog is not None:
                if available[j]:
                    if (
                        identity != future_chain[h - 1]
                        or not isinstance(identity, str)
                        or intervals[identity]["observed"] != row["target_observed"][j]
                    ):
                        raise ValueError(
                            "Available target must retain actual identity and label mask."
                        )
                else:
                    if (
                        type(identity) is not int
                        or identity not in (-1, cutoff_index + h)
                        or future_chain[h - 1] is not None
                        or row["target_observed"][j]
                    ):
                        raise ValueError(
                            "Nominal unavailable index cannot masquerade as observed label/ID."
                        )
                    _native_gap(
                        metadata,
                        catalog,
                        row,
                        absence_refs[j],
                        cutoff_index + h,
                        break_index=break_index,
                    )
                    gap_refs.add(absence_refs[j])
                    continue
            if type(identity) is int and identity == -1:
                ref = absence_refs[j]
                gap = gaps.get(ref, {}) if isinstance(ref, str) else {}
                lo, hi = (
                    gap.get("first_source_interval_index"),
                    gap.get("last_source_interval_index"),
                )
                if (
                    row["target_observed"][j]
                    or gap.get("complete") is not True
                    or any(
                        gap.get(k) != row[k]
                        for k in ("deployment", "site", "archive", "configuration")
                    )
                    or gap.get("reason")
                    not in {
                        "SOURCE_GAP",
                        "MISSING_SOURCE_INTERVAL",
                        "CONFIGURATION_BOUNDARY",
                        "PING_QUARANTINE",
                        "PARTIAL_SOURCE_INTERVAL",
                        "DUPLICATE_ROW",
                        "MISSING_CHANNEL",
                    }
                    or type(lo) is not int
                    or type(hi) is not int
                    or not 0 <= lo <= cutoff_index + h <= hi
                    or not isinstance(gap.get("source_metadata_sha256"), str)
                    or len(gap["source_metadata_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in gap["source_metadata_sha256"])
                ):
                    raise ValueError(
                        "Missing observed target identity or incomplete source-gap proof."
                    )
                if gap["reason"] == "PING_QUARANTINE" and gap.get("quarantined_pings") != [165]:
                    raise ValueError("Missing ping-boundary quarantine explanation.")
                if gap["reason"] == "CONFIGURATION_BOUNDARY" and (
                    not gap.get("next_configuration")
                    or gap["next_configuration"] == row["configuration"]
                ):
                    raise ValueError("Missing actual configuration-boundary explanation.")
                gap_refs.add(ref)
                continue
            if not isinstance(identity, str) or absence_refs[j] is not None:
                raise ValueError("Actual target identity conflicts with structural absence.")
            target = identity_to_stamp.get(identity)
            if (
                target is None
                or target[0] != dep
                or target[2] != 0
                or identity_to_index.get(identity) != cutoff_index + h
                or not timedelta(minutes=55 * h) <= target[1] - cutoff <= timedelta(minutes=65 * h)
            ):
                raise ValueError(
                    "Future target raw identity/timestamp differs from issued horizon."
                )
        used = {v for slots in context for v in slots if isinstance(v, str)} | {
            v for v in targets if isinstance(v, str)
        }
        all_used |= used
        all_used |= {v for v in future_chain if isinstance(v, str)}
        _source_start, source_end, label, end, suffix_start = starts[dep]
        # Bounds for unknown cells are conservative adjacency envelopes, not
        # fabricated exact-hour raw timestamps. Known targets remain mandatory
        # even when numerically masked: an out-of-prefix label cannot be hidden.
        windows = [
            (identity_to_stamp[v][1], identity_to_stamp[v][1])
            if isinstance(v, str)
            else (cutoff + timedelta(minutes=55 * h), cutoff + timedelta(minutes=65 * h))
            for h, v in zip(HORIZONS, targets, strict=True)
        ]
        if all(label <= lo and hi < end for lo, hi in windows) and cutoff < end:
            fit.append(key)
        elif any(type(v) is int for v in targets) and cutoff < end:
            uncertain_prefix.append(key)
        if cutoff >= suffix_start and all(
            identity_to_stamp[v][1] < source_end for v in targets if isinstance(v, str)
        ):
            suffix.append(key)
    if all_used != set(intervals):
        raise ValueError("Registry must exactly identify its declared issuance union.")
    if gap_refs != set(gaps):
        raise ValueError("Source-gap registry must exactly explain its declared absences.")
    by_key = {(r["deployment"], r["row_id"]): r for r in rows}

    def used_ids(keys):
        return {
            v
            for key in keys
            for v in chain(
                chain.from_iterable(by_key[key]["context_ids"]), by_key[key]["target_ids"]
            )
            if isinstance(v, str)
        }

    fitted, held = used_ids(fit), used_ids(suffix)
    if fitted & held or {identity_to_stamp[v] for v in fitted} & {
        identity_to_stamp[v] for v in held
    }:
        raise ValueError("Fit context/targets overlap suffix context/targets.")
    support = []
    counts = {}
    for key in sorted(suffix):
        row = by_key[key]
        stamps = [
            identity_to_stamp[v][1].isoformat() if isinstance(v, str) else None
            for v in row["target_ids"]
        ]
        support.append(
            {
                "deployment": key[0],
                "row_id": key[1],
                "target_ids": row["target_ids"],
                "target_timestamps": stamps,
                "observed": row["target_observed"],
                "target_absence_refs": row.get("target_absence_refs", [None] * 3),
                **(
                    {
                        "issued_configuration": row["configuration"],
                        "target_requested_indices": row["target_requested_indices"],
                        "target_available": [isinstance(v, str) for v in row["target_ids"]],
                    }
                    if catalog is not None
                    else {}
                ),
            }
        )
        for h, stamp, observed in zip(HORIZONS, stamps, row["target_observed"], strict=True):
            if observed:
                count_key = (key[0], h, stamp[:10])
                counts[count_key] = counts.get(count_key, 0) + 1
    eligible = [
        {"deployment": d, "horizon": h, "target_source_date": day, "rows": n}
        for (d, h, day), n in sorted(counts.items())
        if n >= 18
    ]
    complete_suffix = all(
        any(r["deployment"] == d and r["horizon"] == h for r in eligible)
        for d in sources
        for h in HORIZONS
    )
    return {
        "fit_rows": fit,
        "suffix_rows": suffix,
        "fit_interval_ids": sorted(fitted),
        "suffix_interval_ids": sorted(held),
        "boundaries": {d: [t.isoformat() for t in v] for d, v in starts.items()},
        "reserved_suffix_support": support,
        "suffix_support_sha256": sha(json.dumps(support, sort_keys=True).encode()),
        "suffix_eligible_daily_support": eligible,
        "suffix_support_status": "ASSESSABLE_METADATA_SUPPORT"
        if complete_suffix
        else "NOT_ASSESSABLE",
        "source_clock": "nominal_source_calendar_not_verified_UTC",
        "source_gaps": copy.deepcopy(gaps),
        "prefix_absence_boundary_unproven_rows": uncertain_prefix,
        "absence_boundary_policy": "conservative_55_65min_envelope_not_actual_missing_timestamps",
        "configuration_mode": metadata["configuration_mode"],
        "configuration_map": copy.deepcopy(catalog),
        "metadata_only_interval_ids": sorted(all_used - fitted - held),
    }


@lru_cache(maxsize=1)
def required_sources():
    """Static local source closure, including lazy factories; no manifest imports."""
    root = Path(core.__file__).resolve().parents[1]
    pending = [Path(__file__).resolve(), root / "__init__.py"]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        if path.is_relative_to(root):
            package = path.parent
            while package.is_relative_to(root):
                initializer = package / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer)
                package = package.parent
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                names = [name, *(name + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    return sorted(found)


def _backbone(raw, family, band_artifact_version=1):
    """Preserve original config bytes; adapt only the band Config field wrapper."""
    fields = dict(raw)
    if family == "band":
        if fields.pop("architecture", None) != ARCHITECTURES["band"]:
            raise ValueError("Original band config architecture required.")
        if band_artifact_version == 2:
            from marine_echo.training.native_band_replication_ssl import Config

            Config(**raw).validate(correctness_smoke=True)
        elif band_artifact_version != 1 or fields.get("seed") != 7:
            raise ValueError("Original band seed7 remains frozen.")
    elif "architecture" in fields:
        raise ValueError("Legacy/CF config cannot be reinterpreted as band.")
    return core.Config(**fields)


def _validate_backbone(config, backbone):
    c = _backbone(backbone, config.family, config.band_artifact_version)
    if config.family == "band" and c.seed != config.seed:
        raise ValueError("Band parent and prefix seeds must match exactly.")
    if c.method != config.method or c.history != 96:
        raise ValueError("Exact saved backbone method/history differs.")
    d = core.model_dimensions(c)
    expected = (
        {"width": 256, "latent": 128, "blocks": 5, "heads": 4}
        if config.family == "cf"
        else {"width": 192, "latent": 64, "blocks": 4, "heads": 4}
    )
    if (
        any(type(v) is not int or v < 1 for v in d.values())
        or d["width"] % d["heads"]
        or (not config.correctness_smoke and d != expected)
    ):
        raise ValueError("Exact declared backbone dimensions required.")
    return c


def selected_kind(config):
    if config.family == "band" and config.band_artifact_version == 2:
        return REPLICATION_SUPERVISED_KIND if config.method == "direct" else REPLICATION_KIND
    family = "band" if config.family == "band" else "core"
    return SUPERVISED_KINDS[family] if config.method == "direct" else KINDS[family]


def _supervised_ancestry(value, downstream=None):
    """Only genuine fresh direct endpoints; adapted/full-finetuned parents closed."""
    if (
        not isinstance(value, dict)
        or value.get("mode") != "direct_end_to_end"
        or value.get("ssl_only") is not False
        or any(
            k not in value or value[k] is not None
            for k in ("ancestor_encoder_sha256", "ancestor_run_sha256")
        )
    ):
        raise ValueError("Complete fresh direct supervised ancestry required; no adapted parent.")
    updates, step = value.get("supervised_updates"), value.get("selected_supervised_step")
    if type(updates) is not int or type(step) is not int or not 1 <= step <= updates <= 3000:
        raise ValueError("Actual supervised label/selection ancestry required.")
    if downstream is not None and (updates > downstream["updates"] or step % downstream["cadence"]):
        raise ValueError("Parent supervised selection/configuration differs.")
    return copy.deepcopy(value)


def _split_members(split, config):
    """Actual frozen schema; legacy test aliases exist only in private fixtures."""
    schema = split.get("schema_version")
    if schema != "native_acoustic_ssl_v1" and not (schema is None and config.correctness_smoke):
        raise ValueError("Exact native_acoustic_ssl_v1 split schema required.")
    sources = split.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Explicit split source reservations required.")
    allowed = {"train", "development", "final_test"}
    if schema is None and config.correctness_smoke:
        allowed.add("test")
    if any(
        not isinstance(s, dict)
        or s.get("role") not in allowed
        or not s.get("deployment")
        or not s.get("archive_sha256")
        for s in sources
    ):
        raise ValueError("Unknown/missing split roles or identities.")
    roles = {s["role"] for s in sources}
    if "test" in roles and "final_test" in roles:
        raise ValueError("Mixed split test-role aliases are forbidden.")
    members = {s["deployment"]: s for s in sources}
    if len(members) != len(sources):
        raise ValueError("Duplicate/conflicting split deployment identity.")
    archives = {}
    for s in sources:
        prior = archives.setdefault(s["archive_sha256"], s["role"])
        if prior != s["role"]:
            raise ValueError("Split archive fitted/assessment roles overlap.")
    return members


LOADED_SOURCE_HASHES = {str(p): sha(p.read_bytes()) for p in required_sources()}


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit immutable path required.")
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def _distinct(record, manifest, status, scope):
    excluded = {
        IMPLEMENTER_SESSION_ID,
        manifest["implementer_session_id"],
        manifest["coordinator_session_id"],
    }
    reviewer = record.get("reviewer_session_id")
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or reviewer.casefold() in {v.casefold() for v in excluded}
    ):
        raise ValueError("Distinct reviewer required.")
    if any(
        record.get(k) != v
        for k, v in {
            "status": status,
            "scope": scope,
            "role": "prefix_transfer",
            "evidence_kind": manifest["evidence_kind"],
            "implementer_session_id": manifest["implementer_session_id"],
            "coordinator_session_id": manifest["coordinator_session_id"],
        }.items()
    ):
        raise ValueError("Review role/status/evidence/scope/identity differs.")
    if not isinstance(record.get("bindings"), dict) or not record["bindings"]:
        raise ValueError("Nonempty exact bindings required.")
    return record["bindings"]


@dataclass
class Admission:
    manifest: dict
    config: PrefixConfig
    snapshots: dict
    documents: dict
    partition: dict
    identities: dict


def admit(manifest_path, review_path, output_path, *, resume=None):
    """All bindings and ancestry checked BEFORE NPZ/tensor decode or RNG/model setup."""
    manifest_path, review_path = Path(manifest_path).resolve(), Path(review_path).resolve()
    base = manifest_path.parent
    output = Path(output_path).resolve()
    if (output.exists() and resume is None) or (resume is not None and not output.is_dir()):
        raise FileExistsError("New output required; existing run needs explicit resume.")
    if (output / "completion.json").exists() or (output / "inference.pt").exists():
        raise FileExistsError("Completed output is protected; no silent replay overwrite.")
    manifest = _json(manifest_path.read_bytes())
    review = _json(review_path.read_bytes())
    if (
        manifest.get("kind") != "native_prefix_transfer_manifest_v1"
        or manifest.get("role") != "prefix_transfer"
    ):
        raise ValueError("Explicit separately versioned prefix-transfer role required.")
    config = PrefixConfig(**manifest["config"])
    config.validate(manifest["evidence_kind"])
    if manifest["evidence_kind"] not in (EVIDENCE, "REVIEWED_PREFIX_TRANSFER"):
        raise ValueError("Unsupported evidence identity.")
    if manifest.get("suffix_numeric_path") is not None:
        raise ValueError("Suffix numerical loading is a later separately reviewed inference.")
    if config.correctness_smoke and manifest.get("fixture_identity") != EVIDENCE:
        raise ValueError("Private synthetic fixture identity required.")
    if review.get("config") != config.to_dict() or review.get("allowed_cells") != [
        {
            "method": config.method,
            "mode": config.mode,
            "seed": config.seed,
            "prefix_days": config.prefix_days,
            "family": config.family,
        }
    ]:
        raise ValueError("Exact runtime config/method/seed/prefix/mode admission differs.")
    bindings = _distinct(
        review, manifest, "APPROVED_PREFIX_TRANSFER_PREFIT", "native_prefix_transfer_fit"
    )
    snapshots, documents = {}, {}

    def bound(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        if bindings.get(str(path)) != sha(raw):
            raise ValueError(f"Missing/stale exact binding: {path}")
        snapshots[path] = raw
        return raw

    def doc(value, relative=base):
        path = _path(value, relative)
        if path not in documents:
            documents[path] = _json(bound(path))
        return path, documents[path]

    bound(manifest_path)
    for path in required_sources():
        bound(path)
        if sha(snapshots[path]) != LOADED_SOURCE_HASHES[str(path)]:
            raise ValueError("Loaded source differs from its exact admitted bytes.")
    for key in (
        "protocol",
        "split",
        "selection",
        "dependency_lock",
        "supervisor",
        "source_manifest",
    ):
        bound(_path(manifest[key], base))
    _, cfg_doc = doc(manifest["config_path"])
    if cfg_doc != config.to_dict():
        raise ValueError("Frozen config file/runtime mismatch.")
    _, architecture = doc(manifest["backbone_config"])
    method_cfg = _validate_backbone(config, architecture)
    if method_cfg.method != config.method or method_cfg.history != 96:
        raise ValueError("Original selected architecture/method identity differs.")
    dims = core.model_dimensions(method_cfg)
    expected = (
        {"width": 256, "latent": 128, "blocks": 5, "heads": 4}
        if config.family == "cf"
        else {
            "width": 192,
            "latent": 64,
            "blocks": 4,
            "heads": 4,
        }
    )
    if not config.correctness_smoke and dims != expected:
        raise ValueError("Frozen actual backbone dimensions differ.")
    if config.family == "cf":
        for p in core.required_sources(method_cfg):
            bound(p)
    if config.family == "band" and manifest.get("architecture") != ARCHITECTURES["band"]:
        raise ValueError("Explicit band architecture required.")
    _, metadata = doc(manifest["raw_intervals"])
    if not config.correctness_smoke and metadata.get("kind") != "native_prefix_raw_intervals_v2":
        raise ValueError("Real native prefix fitting requires the versioned configuration map.")
    partition = partitions(metadata, config.prefix_days)
    if metadata.get("configuration_map") is not None:
        _, configuration_receipt = doc(manifest["configuration_map"])
        if configuration_receipt != metadata["configuration_map"]:
            raise ValueError("Separately bound native configuration map differs.")
        for dep, path in configuration_receipt["source_metadata"].items():
            actual_metadata = bound(_path(path, base))
            entry = configuration_receipt["sources"][dep]
            if sha(actual_metadata) != entry["source_metadata_sha256"]:
                raise ValueError("Actual native source configuration metadata identity differs.")
            source_evidence = _json(actual_metadata)
            if (
                source_evidence.get("kind") != "native_prefix_source_configuration_metadata_v1"
                or source_evidence.get("complete") is not True
                or source_evidence.get("source") != metadata["sources"][dep]
                or source_evidence.get("configurations") != entry["configurations"]
                or source_evidence.get("segments") != entry["segments"]
            ):
                raise ValueError(
                    "Derived configuration metadata differs from actual bound source metadata."
                )
    if metadata.get("source_gaps"):
        _, gap_receipt = doc(manifest["source_gaps"])
        if (
            gap_receipt.get("kind") != "native_prefix_source_gaps_v1"
            or gap_receipt.get("complete") is not True
            or gap_receipt.get("gaps") != metadata["source_gaps"]
        ):
            raise ValueError("Separately bound reader source-gap explanation differs.")
        evidence = gap_receipt.get("source_metadata", {})
        if not isinstance(evidence, dict) or set(evidence) != set(metadata["source_gaps"]):
            raise ValueError("Every source gap requires bound nonnumerical source evidence.")
        for ref, path in evidence.items():
            if (
                sha(bound(_path(path, base)))
                != metadata["source_gaps"][ref]["source_metadata_sha256"]
            ):
                raise ValueError("Source-gap metadata identity differs.")
    _, source_record = doc(manifest["source_manifest"])
    if source_record.get("sources") != metadata["sources"] or source_record.get(
        "quarantined_pings"
    ) != [165]:
        raise ValueError("Frozen source dates/configuration/165 quarantine differ.")
    split_path, split = doc(manifest["split"])
    split_members = _split_members(split, config)
    for dep, source in metadata["sources"].items():
        record = split_members.get(dep, {})
        if (
            record.get("role") not in ("test", "final_test")
            or record.get("archive_sha256") != source["archive"]
        ):
            raise ValueError("Frozen reserved deployment/archive split membership differs.")
    for key, role, rows in (
        ("prefix_cohort", "prefix", partition["fit_rows"]),
        ("dev_cohort", "development", None),
    ):
        _, cohort = doc(manifest[key])
        if cohort.get("role") != role or cohort.get("complete") is not True:
            raise ValueError("Exact complete cohort role required.")
        keys = [tuple(v) for v in cohort["rows"]]
        if config.correctness_smoke and any(EVIDENCE not in v[1] for v in keys):
            raise ValueError("Private synthetic issuance identities required; no science shortcut.")
        if len(keys) != len(set(keys)) or (rows is not None and set(keys) != set(rows)):
            raise ValueError("Cohort exact reserved row set differs.")
        if role == "development" and (
            cohort.get("native_bounds_m") != [0, 225]
            or any(v[0] in metadata["sources"] for v in keys)
        ):
            raise ValueError("Selection is original DEV0–225, never adapted-site outcomes.")
        if role == "development" and any(
            split_members.get(v[0], {}).get("role") != "development" for v in keys
        ):
            raise ValueError("Original development split membership differs.")
    _, selection = doc(manifest["selection"])
    if (
        selection.get("role") != "original_development"
        or selection.get("cohort_sha256") != sha(snapshots[_path(manifest["dev_cohort"], base)])
        or selection.get("opportunities")
        != list(range(config.cadence, config.updates + 1, config.cadence))
        or selection.get("floor") != 18
        or selection.get("aggregation") != "equal_target_source_date_then_horizon_then_deployment"
    ):
        raise ValueError("Frozen original-development selection recipe differs.")
    _, statistics = doc(manifest["scalers"])
    _scalers(statistics)
    reserved = metadata["sources"]
    forbidden = {k: {s[k] for s in reserved.values()} for k in ("site", "deployment", "archive")}
    artifacts, visiting, complete = set(), set(), set()
    ancestor_inputs, ancestor_members = {}, set()

    def ancestry(value, relative=base):
        path, node = doc(value, relative)
        if path in visiting:
            raise ValueError("Recursive ancestry cycle.")
        if path in complete:
            return
        visiting.add(path)
        if (
            node.get("kind") != "native_prefix_train_ancestry_v1"
            or node.get("complete") is not True
            or node.get("historical_initial_weights") is not False
        ):
            raise ValueError("Unknown/incomplete/historical initial fitted ancestry.")
        if node.get("role") != "train" or not node.get("inputs"):
            raise ValueError("Original TRAIN-only fitted ancestry required.")
        for inp in node["inputs"]:
            _, cohort = doc(inp["cohort"], path.parent)
            if (
                cohort.get("role") != "train"
                or cohort.get("complete") is not True
                or cohort.get("members") != inp.get("members")
                or not inp.get("members")
            ):
                raise ValueError("Exact ancestor input/cohort membership required.")
            input_raw = bound(_path(inp["npz"], path.parent))
            ancestor_inputs[sha(input_raw)] = _path(inp["npz"], path.parent)
            for member in inp["members"]:
                if any(not member.get(k) or member[k] in forbidden[k] for k in forbidden):
                    raise ValueError("Local ancestor overlaps reserved deployment/site/archive.")
                split_member = split_members.get(member["deployment"], {})
                if (
                    split_member.get("role") != "train"
                    or split_member.get("archive_sha256") != member["archive"]
                ):
                    raise ValueError("Ancestor exact TRAIN split membership differs.")
                ancestor_members.add((member["deployment"], member["archive"]))
            _, raw_receipt = doc(inp["raw_intervals"], path.parent)
            if (
                raw_receipt.get("kind") != "native_prefix_train_interval_receipt_v1"
                or raw_receipt.get("complete") is not True
                or raw_receipt.get("role") != "train"
                or raw_receipt.get("members") != inp["members"]
                or raw_receipt.get("source_npz_sha256") != sha(input_raw)
                or raw_receipt.get("cohort_sha256")
                != sha(snapshots[_path(inp["cohort"], path.parent)])
                or raw_receipt.get("split_sha256") != sha(snapshots[split_path])
            ):
                raise ValueError(
                    "Complete TRAIN raw-interval/source/scaler ancestry receipt required."
                )
        if not node.get("artifacts"):
            raise ValueError("Ancestry artifact identities missing.")
        for item in node["artifacts"]:
            if sha(bound(_path(item["path"], path.parent))) != item["sha256"]:
                raise ValueError("Ancestor artifact identity differs.")
            artifacts.add(item["sha256"])
        for parent in node["parents"]:
            ancestry(parent, path.parent)
        visiting.remove(path)
        complete.add(path)

    ancestry(manifest["scaler_ancestry"])
    if sha(snapshots[_path(manifest["scalers"], base)]) not in artifacts:
        raise ValueError("Scalers lack exact original TRAIN fitted ancestry.")
    if config.mode == "frozen_readout":
        ancestry(manifest["encoder_ancestry"])
        raw = bound(_path(manifest["encoder"], base))
        if sha(raw) not in artifacts:
            raise ValueError("Selected encoder absent from recursive TRAIN ancestry.")
        _, parent_run = doc(manifest["parent_run"])
        parent_inference = bound(_path(manifest["parent_inference"], base))
        parent_membership = bound(_path(manifest["parent_membership"], base))
        parent_backbone = (
            parent_run.get("core_config") if config.method == "direct" else parent_run.get("config")
        )
        if (
            parent_backbone != architecture
            or parent_run.get("inference_sha256") != sha(parent_inference)
            or parent_run.get("membership_sha256") != sha(parent_membership)
            or sha(parent_inference) not in artifacts
            or sha(parent_membership) not in artifacts
        ):
            raise ValueError("Selected parent run configuration/scalers differ.")
        if config.method == "direct":
            _, original_config = doc(manifest["parent_config"])
            review_path, original_review = doc(manifest["parent_review"])
            _, original_input_paths = doc(manifest["parent_inputs"])
            if config.family == "band" and config.band_artifact_version == 2:
                from marine_echo.training import native_band_replication_ssl as original_core
                from marine_echo.training.native_band_replication_downstream import (
                    DownstreamConfig,
                    RunInputs,
                    required_paths,
                )
            elif config.family == "band":
                from marine_echo.training import native_band_ssl as original_core
                from marine_echo.training.native_band_downstream import (
                    DownstreamConfig,
                    RunInputs,
                    required_paths,
                )
            else:
                from marine_echo.training import native_ssl as original_core
                from marine_echo.training.native_downstream import (
                    DownstreamConfig,
                    RunInputs,
                    required_paths,
                )
            original_inputs = RunInputs(
                **{
                    k: _path(v, base) if v is not None else None
                    for k, v in original_input_paths.items()
                }
            )
            if (
                original_inputs.encoder is not None
                or original_inputs.ancestor_review is not None
                or original_inputs.ancestor_config is not None
                or original_inputs.config != _path(manifest["parent_config"], base)
                or original_inputs.review != review_path
                or original_inputs.dev != _path(manifest["dev_npz"], base)
                or original_inputs.split != split_path
                or sha(bound(original_inputs.train)) not in ancestor_inputs
            ):
                raise ValueError("Original direct runtime input/parent ancestry identity differs.")
            downstream = DownstreamConfig(**original_config)
            downstream.validate(correctness_smoke=config.correctness_smoke)
            original_evidence = (
                EVIDENCE if config.correctness_smoke else "REAL_TRAIN_DEVELOPMENT_FIT"
            )
            reviewer = original_review.get("reviewer_session_id")
            excluded = {
                IMPLEMENTER_SESSION_ID,
                manifest["implementer_session_id"],
                manifest["coordinator_session_id"],
                original_review.get("implementer_session_id"),
            }
            if (
                not isinstance(reviewer, str)
                or not reviewer.strip()
                or reviewer.casefold() in {v.casefold() for v in excluded if isinstance(v, str)}
                or not original_review.get("implementer_session_id")
                or original_review.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
                or "direct" not in original_review.get("allowed_methods", [])
                or "direct_end_to_end" not in original_review.get("allowed_modes", [])
                or parent_run.get("status") != "COMPLETED"
                or parent_run.get("evidence_kind") != original_evidence
                or parent_run.get("test_access") != "NOT_RUN"
                or parent_run.get("mode") != "direct_end_to_end"
                or original_config != parent_run.get("config")
                or (downstream.method, downstream.mode, downstream.seed, downstream.history)
                != ("direct", "direct_end_to_end", method_cfg.seed, 96)
                or parent_run.get("review_sha256") != sha(snapshots[review_path])
                or parent_run.get("reviewer_session_id") != reviewer
                or parent_run.get("selected_encoder_sha256") != sha(raw)
            ):
                raise ValueError(
                    "Exact original direct run/config/review/encoder identity required."
                )
            lineage = _supervised_ancestry(parent_run.get("supervised_ancestry"), original_config)
            if any(
                parent_run.get(k) != lineage[k]
                for k in ("supervised_updates", "selected_supervised_step")
            ):
                raise ValueError("Original direct supervised label ancestry differs.")
            original_bindings = parent_run.get("bindings")
            if not isinstance(original_bindings, dict) or not original_bindings:
                raise ValueError("Original direct source/data bindings required.")
            original_sources = manifest.get("parent_source_snapshots", {})
            original_bound = {}
            for p, expected_hash in original_bindings.items():
                if original_review.get("bindings", {}).get(p) != expected_hash:
                    raise ValueError("Original direct source/config review mismatch.")
                actual_path = _path(original_sources.get(p, p), base)
                if sha(bound(actual_path)) != expected_hash:
                    raise ValueError("Original direct source snapshot/data identity differs.")
                original_bound[p] = expected_hash
            required_original = [
                *ancestor_inputs.values(),
                _path(manifest["dev_npz"], base),
                split_path,
                _path(manifest["parent_config"], base),
                Path(original_core.__file__).with_name(
                    (
                        "native_band_replication_downstream.py"
                        if config.band_artifact_version == 2
                        else "native_band_downstream.py"
                    )
                    if config.family == "band"
                    else "native_downstream.py"
                ),
                *original_core.required_sources(original_core.Config(**architecture)),
            ]
            required_original.extend(
                required_paths(original_inputs, original_core.Config(**architecture))
            )
            if any(str(p) not in original_bound for p in required_original):
                raise ValueError("Original direct review misses runtime/source/TRAIN lineage.")
            if (
                original_review.get("train_npz_sha256") not in ancestor_inputs
                or original_review.get("dev_npz_sha256")
                != sha(bound(_path(manifest["dev_npz"], base)))
                or original_review.get("split_sha256") != sha(snapshots[split_path])
            ):
                raise ValueError("Original direct TRAIN/development/split ancestry differs.")
            if (
                config.family == "band"
                and config.band_artifact_version == 2
                and (
                    downstream.seed not in original_review.get("allowed_seeds", [])
                    or original_review.get("architecture") != ARCHITECTURES["band"]
                    or original_review.get("evidence_kind") != original_evidence
                    or set(original_review.get("allowed_roles", [])) != {"train", "development"}
                )
            ):
                raise ValueError(
                    "Exact replication direct parent seed/architecture/role review required."
                )
            membership = _json(parent_membership)
            rows, deployments, archives = (
                membership.get(k)
                for k in ("train_row_ids", "train_deployments", "train_archive_sha256")
            )
            if (
                not isinstance(rows, list)
                or not rows
                or not isinstance(deployments, list)
                or not isinstance(archives, list)
                or len(rows) != len(deployments)
                or len(rows) != len(archives)
                or any(not isinstance(r, str) or not r for r in rows)
                or len(set(zip(deployments, rows, strict=True))) != len(rows)
                or any(
                    (d, a) not in ancestor_members
                    for d, a in zip(deployments, archives, strict=True)
                )
            ):
                raise ValueError("Complete actual direct TRAIN membership required.")
            pool, sequence = membership.get("supervised_indices"), membership.get("sequence")
            if (
                not isinstance(pool, list)
                or not pool
                or len(set(pool)) != len(pool)
                or any(type(i) is not int or not 0 <= i < len(rows) for i in pool)
                or not isinstance(sequence, list)
                or len(sequence) != lineage["supervised_updates"]
                or any(
                    not e.get("indices")
                    or any(type(i) is not int or i not in pool for i in e["indices"])
                    for e in sequence
                )
                or membership.get("sequence_sha256")
                != sha(json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode())
            ):
                raise ValueError("Direct observed-label sampling membership is incomplete.")
            for index, entry in enumerate(sequence):
                ids = entry["indices"]
                if (
                    entry.get("step") != index
                    or len(ids) != min(downstream.batch_size, len(pool))
                    or len(set(ids)) != len(ids)
                    or entry.get("row_ids") != [rows[i] for i in ids]
                    or entry.get("sha256") != core.sequence_hash(np.asarray(ids, dtype=np.int64))
                ):
                    raise ValueError("Actual direct sampling row/index/step/hash receipt differs.")
        elif config.family == "band" and config.band_artifact_version == 2:
            from marine_echo.training import native_band_replication_ssl as original_core

            parent_config_path, original_config = doc(manifest["parent_config"])
            review_path, original_review = doc(manifest["parent_review"])
            _, original_inputs = doc(manifest["parent_inputs"])
            if set(original_inputs) != {"train", "dev", "split", "protocol", "config", "review"}:
                raise ValueError("Exact replication SSL runtime paths required.")
            input_paths = {k: _path(v, base) for k, v in original_inputs.items()}
            if (
                input_paths["dev"] != _path(manifest["dev_npz"], base)
                or input_paths["split"] != split_path
                or input_paths["config"] != parent_config_path
                or input_paths["review"] != review_path
                or sha(bound(input_paths["train"])) not in ancestor_inputs
                or original_config != architecture
            ):
                raise ValueError("Replication SSL runtime/config/TRAIN lineage differs.")
            parent_cfg = original_core.Config(**original_config)
            parent_cfg.validate(correctness_smoke=config.correctness_smoke)
            original_evidence = (
                EVIDENCE if config.correctness_smoke else "REAL_TRAIN_DEVELOPMENT_FIT"
            )
            reviewer = original_review.get("reviewer_session_id")
            excluded = {
                IMPLEMENTER_SESSION_ID,
                manifest["implementer_session_id"],
                manifest["coordinator_session_id"],
                original_review.get("implementer_session_id"),
                original_review.get("root_coordinator_session_id"),
            }
            if (
                not isinstance(reviewer, str)
                or not reviewer.strip()
                or reviewer.casefold() in {v.casefold() for v in excluded if isinstance(v, str)}
                or not original_review.get("root_coordinator_session_id")
                or original_review.get("status") != "APPROVED_PREFIT"
                or original_review.get("architecture") != ARCHITECTURES["band"]
                or original_review.get("evidence_kind") != original_evidence
                or set(original_review.get("allowed_roles", [])) != {"train", "development"}
                or config.seed not in original_review.get("allowed_seeds", [])
                or config.method not in original_review.get("allowed_methods", [])
                or parent_run.get("status") != "COMPLETED"
                or parent_run.get("evidence_kind") != original_evidence
                or parent_run.get("test_access") != "NOT_RUN"
                or parent_run.get("architecture") != ARCHITECTURES["band"]
                or parent_run.get("review_sha256") != sha(snapshots[review_path])
            ):
                raise ValueError(
                    "Genuine distinct completed replication SSL parent review required."
                )
            original_bindings = parent_run.get("bindings")
            if not isinstance(original_bindings, dict) or not original_bindings:
                raise ValueError("Replication SSL source/data/config bindings required.")
            original_sources = manifest.get("parent_source_snapshots", {})
            for p, h in original_bindings.items():
                if (
                    original_review.get("bindings", {}).get(p) != h
                    or sha(bound(_path(original_sources.get(p, p), base))) != h
                ):
                    raise ValueError("Replication SSL source/parent review identity differs.")
            required_original = [
                *(p for k, p in input_paths.items() if k != "review"),
                *original_core.required_sources(parent_cfg),
            ]
            if any(str(p) not in original_bindings for p in required_original):
                raise ValueError("Replication SSL review misses exact source/runtime closure.")
            for key, name in (
                ("train", "train_npz_sha256"),
                ("dev", "dev_npz_sha256"),
                ("split", "split_sha256"),
                ("protocol", "protocol_sha256"),
            ):
                if original_review.get(name) != sha(bound(input_paths[key])):
                    raise ValueError("Replication SSL original TRAIN/dev/split/protocol differs.")
            membership = _json(parent_membership)
            rows = membership.get("train_row_ids")
            deployments = membership.get("train_deployments")
            archives = membership.get("train_archive_sha256")
            if (
                not isinstance(rows, list)
                or not rows
                or not isinstance(deployments, list)
                or not isinstance(archives, list)
                or len(rows) != len(deployments)
                or len(rows) != len(archives)
                or any(not isinstance(r, str) or not r for r in rows)
                or len(set(zip(deployments, rows, strict=True))) != len(rows)
                or any(
                    (d, a) not in ancestor_members
                    for d, a in zip(deployments, archives, strict=True)
                )
                or not isinstance(membership.get("sequence"), list)
                or any(
                    not isinstance(membership.get(k), list)
                    or not membership[k]
                    or len(set(membership[k])) != len(membership[k])
                    or any(type(i) is not int or not 0 <= i < len(rows) for i in membership[k])
                    for k in ("ssl_eligible_indices", "supervised_indices")
                )
            ):
                raise ValueError("Complete replication SSL original TRAIN membership required.")
            sequence = membership["sequence"]
            counts = {"pretrain": 0, "readout": 0}
            for entry in sequence:
                phase, ids = entry.get("phase"), entry.get("context_indices")
                pool = membership[
                    "ssl_eligible_indices" if phase == "pretrain" else "supervised_indices"
                ]
                if (
                    phase not in counts
                    or not isinstance(ids, list)
                    or len(ids) != min(parent_cfg.batch_size, len(pool))
                    or len(set(ids)) != len(ids)
                    or any(type(i) is not int or i not in pool for i in ids)
                    or type(entry.get("step")) is not int
                    or entry["step"] < 0
                    or entry.get("context_sha256")
                    != core.sequence_hash(np.asarray(ids, dtype=np.int64))
                ):
                    raise ValueError("Replication SSL sampling phase/index/hash receipt differs.")
                if phase == "pretrain":
                    targets = entry.get("target_indices")
                    if (
                        not isinstance(targets, list)
                        or sorted(targets) != sorted(ids)
                        or (config.method != "permuted_ssl" and targets != ids)
                        or entry.get("target_multiset_sha256")
                        != core.sequence_hash(np.asarray(sorted(ids), dtype=np.int64))
                    ):
                        raise ValueError("Replication SSL pairing/multiset receipt differs.")
                counts[phase] += 1
            if (
                type(parent_run.get("pretrain_steps")) is not int
                or type(parent_run.get("readout_steps_all_probes")) is not int
                or counts
                != {
                    "pretrain": parent_run["pretrain_steps"],
                    "readout": parent_run["readout_steps_all_probes"],
                }
                or parent_run.get("optimizer_steps_total") != sum(counts.values())
                or counts["pretrain"] > parent_cfg.pretrain_updates
                or (config.method == "random_frozen" and counts["pretrain"] != 0)
            ):
                raise ValueError("Actual replication SSL update/membership accounting differs.")
    elif manifest.get("encoder") is not None:
        raise ValueError("Scratch must have no inherited encoder.")
    for key in ("prefix_npz", "dev_npz"):
        bound(_path(manifest[key], base))
    _, numeric = doc(manifest["numeric_access_review"])
    numeric_bindings = _distinct(
        numeric, manifest, "APPROVED_PREFIX_TRANSFER_NUMERIC_ACCESS", "prefix_only_numeric_fit"
    )
    if numeric.get("allowed_uses") != [
        "prefix_fit",
        "original_development_selection",
        "suffix_metadata_only",
    ]:
        raise ValueError("Separate frozen prefix/suffix numeric-access scope required.")
    if numeric.get("config") != config.to_dict():
        raise ValueError("Numeric review exact cell/config differs.")
    for p, raw in snapshots.items():
        if p != _path(manifest["numeric_access_review"], base) and numeric_bindings.get(
            str(p)
        ) != sha(raw):
            raise ValueError("Numeric provenance missing/stale source/model/partition binding.")
    identities = {str(p): sha(raw) for p, raw in snapshots.items()}
    if resume is not None:
        bound(Path(resume).resolve())
    return Admission(manifest, config, snapshots, documents, partition, identities)


def _strict_state(module, state, *, assign=True):
    expected = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Exact encoder/head parameters and buffers required.")
    for k, t in expected.items():
        v = state[k]
        if (
            not isinstance(v, torch.Tensor)
            or v.device.type != "cpu"
            or v.layout != torch.strided
            or v.shape != t.shape
            or v.dtype != t.dtype
            or not torch.isfinite(v).all()
        ):
            raise ValueError("Unsafe/incompatible/nonfinite tensor state.")
    module.load_state_dict(state, strict=True, assign=assign)


class PrefixModel(nn.Module):
    def __init__(self, backbone_config, family, band_artifact_version=1):
        super().__init__()
        c = _backbone(backbone_config, family, band_artifact_version)
        d = core.model_dimensions(c)
        self.encoder = (
            CFTemporalEncoder(d["width"], d["latent"], d["blocks"])
            if family == "cf"
            else (NativeBandEncoder if family == "band" else SharedTemporalEncoder)(
                d["width"], d["latent"], d["blocks"], d["heads"]
            )
        )
        self.head = QueryHead(d["latent"], 5, d["latent"])

    def forecast(self, x, observed, metadata, query, *, frozen=True):
        if frozen:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder.encode(x, observed, metadata)
        else:
            z = self.encoder.encode(x, observed, metadata)
        return self.head(z, query).sort(-1).values


def prepare_model(config, backbone, selected, scalers, *, parent_bindings=None):
    """Fresh head; only scratch constructs random encoder. No legacy fit initializer."""
    _validate_backbone(config, backbone)
    if config.mode == "frozen_readout":
        expected_kind = selected_kind(config)
        if (
            selected is None
            or selected.get("kind") != expected_kind
            or selected.get("config") != backbone
            or selected.get("scalers") != scalers
        ):
            raise ValueError("Exact selected kind/config/TRAIN scalers required.")
        if parent_bindings is not None and selected.get("bindings") != parent_bindings:
            raise ValueError("Selected original source ancestry differs from parent run.")
        if config.family == "band" and selected.get("architecture") != ARCHITECTURES["band"]:
            raise ValueError("Band selected architecture mismatch.")
        if config.method == "direct":
            _supervised_ancestry(selected.get("supervised_ancestry"))
        elif selected.get("supervised_ancestry") is not None:
            raise ValueError("SSL/control parent cannot hide supervised feature ancestry.")
        with torch.device("meta"):
            model = PrefixModel(backbone, config.family, config.band_artifact_version).float()
        _strict_state(model.encoder, selected.get("encoder"))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 100000)
            d = core.model_dimensions(
                _backbone(backbone, config.family, config.band_artifact_version)
            )["latent"]
            model.head = QueryHead(d, 5, d)
    else:
        if selected is not None:
            raise ValueError("Scratch may not load selected weights.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = PrefixModel(backbone, config.family, config.band_artifact_version).float()
            torch.manual_seed(config.seed + 100000)
            d = core.model_dimensions(
                _backbone(backbone, config.family, config.band_artifact_version)
            )["latent"]
            model.head = QueryHead(d, 5, d)
    model.to(config.device)
    model.encoder.requires_grad_(config.mode == "scratch_direct")
    model.encoder.eval()
    model.feature_ancestor = {
        "selected_kind": selected["kind"] if selected is not None else None,
        "training_kind": TRAINING_KINDS[config.method]
        if selected is not None
        else "fresh_scratch_supervised_encoder",
        "supervised_ancestry": copy.deepcopy(selected.get("supervised_ancestry"))
        if selected is not None
        else None,
        "original_parent_bindings": copy.deepcopy(selected.get("bindings"))
        if selected is not None
        else None,
        "original_train_scalers": copy.deepcopy(scalers),
        "parent_head_reused": False,
        "fresh_head_seed": config.seed + 100000,
    }
    return model


def _decode(admission, key, role):
    m, base = admission.manifest, Path(admission.manifest["_base"])
    with np.load(io.BytesIO(admission.snapshots[_path(m[key], base)]), allow_pickle=False) as z:
        keys = (
            "x",
            "context_observed",
            "metadata",
            "query",
            "targets",
            "target_observed",
            "row_id",
            "deployment",
            "target_dates",
            "corpus_role",
            "evidence_kind",
        )
        data = {k: z[k].copy() for k in keys}
        if admission.config.correctness_smoke and str(z["fixture_identity"].item()) != EVIDENCE:
            raise ValueError("Synthetic numerical fixture marker missing.")
    if (
        str(data.pop("corpus_role").item()) != role
        or str(data.pop("evidence_kind").item()) != m["evidence_kind"]
    ):
        raise ValueError("Numeric corpus role/evidence mismatch.")
    n = len(data["x"])
    _inputs(data["x"], data["context_observed"], data["metadata"], 96)
    expected_lower = 230 if role == "prefix" else 225
    if not np.allclose(data["metadata"][:, 0, 3:5] * 250, [0, expected_lower], atol=1e-5, rtol=0):
        raise ValueError("Primary native prefix230/original DEV225 geometry differs.")
    query = data["query"]
    if (
        query.shape != (n, 3, 10)
        or not np.isfinite(query).all()
        or not np.allclose(query[..., :9], data["metadata"][:, :1, :9])
        or not np.array_equal(query[..., 9], np.broadcast_to(HORIZONS, (n, 3)))
    ):
        raise ValueError("Only native issued1/3/6 query supported.")
    y, mask = data["targets"], data["target_observed"]
    if (
        y.shape != (n, 3)
        or y.dtype.kind != "f"
        or mask.shape != y.shape
        or mask.dtype != np.bool_
        or not np.isfinite(y[mask]).all()
    ):
        raise ValueError("Finite observed targets and Boolean[N,3] support required.")
    if data["target_dates"].shape != y.shape or data["target_dates"].dtype.kind != "U":
        raise ValueError("Source-calendar target dates required.")
    for value in set(data["target_dates"][mask].tolist()):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Observed labels require exact source-calendar dates.")
    cohort = admission.documents[
        _path(m["prefix_cohort" if role == "prefix" else "dev_cohort"], base)
    ]
    rows = list(zip(data["deployment"].tolist(), data["row_id"].tolist(), strict=True))
    expected = [tuple(v) for v in cohort["rows"]]
    if len(set(rows)) != n or set(rows) != set(expected):
        raise ValueError("Exact numeric row/cohort identities required; no intersection.")
    if role == "prefix":
        metadata = admission.documents[_path(m["raw_intervals"], base)]
        declarations = {(r["deployment"], r["row_id"]): r for r in metadata["rows"]}
        for i, row in enumerate(rows):
            declared = declarations[row]
            if data["target_observed"][i].tolist() != declared["target_observed"]:
                raise ValueError("Numeric target masks differ from root-bound raw support.")
            if metadata.get("configuration_map") is not None:
                cfg = metadata["configuration_map"]["sources"][row[0]]["configurations"][
                    declared["configuration"]
                ]
                expected_bounds = cfg["channel_bounds_m"]
                if not np.array_equal(
                    data["context_observed"][i], np.asarray(declared["context_observed"], bool)
                ) or not np.array_equal(
                    data["metadata"][i, :, 6],
                    [p != "UNKNOWN" for p in cfg["processing_id_or_unknown"]],
                ):
                    raise ValueError(
                        "Numerical observed context/processing conflicts with native configuration metadata."
                    )
            else:
                expected_bounds = metadata["sources"][row[0]]["channel_bounds_m"]
            if not np.allclose(
                data["metadata"][i, :, 3:5] * 250, expected_bounds, atol=1e-5, rtol=0
            ):
                raise ValueError(
                    "Numerical native geometry conflicts with raw interval provenance."
                )
            dates = [
                metadata["intervals"][v]["timestamp"][:10] if isinstance(v, str) else ""
                for v in declared["target_ids"]
            ]
            if data["target_dates"][i].tolist() != dates:
                raise ValueError("Native date proxies conflict with actual raw timestamps.")
    # Canonical issuance order yields identical sample sequence across archives/methods.
    order = np.asarray(sorted(range(n), key=lambda i: rows[i]))
    return {k: v[order] for k, v in data.items()}


def sample_indices(size, config, step):
    return core.batch_indices(np.arange(size), config.batch_size, config.seed, "readout", step)


def _batch(data, indices, scalers, device, *, labels):
    x, observed = data["x"][indices], data["context_observed"][indices]
    arrays = {
        "x": scalers.channels(x, observed),
        "observed": observed,
        "metadata": data["metadata"][indices],
        "query": data["query"][indices],
    }
    if labels:
        arrays.update(
            y=scalers.targets(data["targets"][indices], data["target_observed"][indices]),
            y_observed=data["target_observed"][indices],
        )
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def encode_checkpoint(value):
    stream = io.BytesIO()
    torch.save(_safe_cpu(value), stream)
    return stream.getvalue()


def _safe_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _safe_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_safe_cpu(v) for v in value)
    return copy.deepcopy(value)


def decode_checkpoint(raw):
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def predict(model, data, scalers, config):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(data["x"]), config.batch_size):
            b = _batch(
                data,
                np.arange(start, min(start + config.batch_size, len(data["x"]))),
                scalers,
                config.device,
                labels=False,
            )
            p = model.forecast(**b).cpu().numpy()
            result.append(
                p * scalers.target_std[None, :, None] + scalers.target_mean[None, :, None]
            )
    return np.concatenate(result)


class _Trajectory:
    """Private trajectory: real construction requires a verified admission.

    Synthetic constructor is CPU-only and visibly guarded; no public data loader
    has a synthetic shortcut. Full RNG, samples, optimizer and DEV state resume.
    """

    def __init__(
        self,
        config,
        backbone,
        selected,
        statistics,
        prefix,
        dev,
        identities,
        *,
        admission=None,
        parent_bindings=None,
    ):
        config.validate(EVIDENCE if config.correctness_smoke else "REVIEWED_PREFIX_TRANSFER")
        if not config.correctness_smoke and (
            not isinstance(admission, Admission)
            or admission.config != config
            or admission.identities != identities
        ):
            raise ValueError("Real trajectory requires exact admitted fit.")
        self.config, self.backbone, self.statistics = config, backbone, copy.deepcopy(statistics)
        self.prefix, self.dev, self.identities = prefix, dev, identities
        self.scalers = _scalers(statistics)
        self.model = prepare_model(
            config, backbone, selected, statistics, parent_bindings=parent_bindings
        )
        self.initial_encoder = core.cpu_state(self.model.encoder)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = core.schedule(self.optimizer, config.updates)
        self.step, self.samples, self.candidates = 0, [], []
        self.best_score, self.selected_step, self.selected_state = None, None, None
        self.elapsed = 0.0

    def advance(self, stop=None):
        stop = self.config.updates if stop is None else stop
        if not self.step <= stop <= self.config.updates:
            raise ValueError("Invalid continuation limit.")
        start = time.perf_counter()
        for step in range(self.step, stop):
            self.model.head.train()
            self.model.encoder.train(self.config.mode == "scratch_direct")
            indices = sample_indices(len(self.prefix["x"]), self.config, step)
            self.samples.append(indices.tolist())
            b = _batch(self.prefix, indices, self.scalers, self.config.device, labels=True)
            self.optimizer.zero_grad(set_to_none=True)
            p = self.model.forecast(
                b["x"],
                b["observed"],
                b["metadata"],
                b["query"],
                frozen=self.config.mode == "frozen_readout",
            )
            # Targets absent/masked do not enter residuals or gradients.
            loss = core.pinball(p, b["y"], b["y_observed"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised objective.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), self.config.clip
            )
            self.optimizer.step()
            self.scheduler.step()
            self.step = step + 1
            if self.step % self.config.cadence == 0:
                predictions = predict(self.model, self.dev, self.scalers, self.config)
                metrics = native_scores(
                    predictions,
                    self.dev["targets"],
                    self.dev["target_observed"],
                    self.dev["target_dates"],
                    self.dev["deployment"],
                    minimum_daily_rows=18,
                )
                score = metrics["primary_pinball_db"]
                self.candidates.append(
                    {"step": self.step, "role": "original_development", "metrics": metrics}
                )
                if score is not None and (self.best_score is None or score < self.best_score):
                    self.best_score, self.selected_step = score, self.step
                    self.selected_state = core.cpu_state(self.model)
            if self.config.mode == "frozen_readout" and any(
                not torch.equal(v, self.model.encoder.state_dict()[k].cpu())
                for k, v in self.initial_encoder.items()
            ):
                raise RuntimeError("Frozen encoder parameters/buffers changed.")
        self.elapsed += time.perf_counter() - start

    def checkpoint(self):
        return {
            "kind": "native_prefix_transfer_resume_v1",
            "config": self.config.to_dict(),
            "backbone_config": self.backbone,
            "architecture": ARCHITECTURES[self.config.family],
            "scalers": self.statistics,
            "identities": self.identities,
            "feature_ancestor": copy.deepcopy(self.model.feature_ancestor),
            "model": core.cpu_state(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": core.rng_state(),
            "initial_encoder": self.initial_encoder,
            "step": self.step,
            "samples": self.samples,
            "candidates": self.candidates,
            "selected_step": self.selected_step,
            "selected_state": self.selected_state,
            "best_score": self.best_score,
            "elapsed_seconds": self.elapsed,
        }

    def restore(self, checkpoint):
        if checkpoint.get("kind") != "native_prefix_transfer_resume_v1" or any(
            checkpoint.get(k) != v
            for k, v in {
                "config": self.config.to_dict(),
                "backbone_config": self.backbone,
                "architecture": ARCHITECTURES[self.config.family],
                "scalers": self.statistics,
                "identities": self.identities,
                "feature_ancestor": self.model.feature_ancestor,
            }.items()
        ):
            raise ValueError("Exact safe resume identities/config/device/scalers required.")
        step = checkpoint["step"]
        if (
            type(step) is not int
            or not 0 <= step <= self.config.updates
            or checkpoint["samples"]
            != [sample_indices(len(self.prefix["x"]), self.config, s).tolist() for s in range(step)]
        ):
            raise ValueError("Resume step/sample sequence differs.")
        if set(checkpoint["initial_encoder"]) != set(self.initial_encoder) or any(
            not torch.equal(v, checkpoint["initial_encoder"][k])
            for k, v in self.initial_encoder.items()
        ):
            raise ValueError("Original frozen/random initializer ancestry differs on resume.")
        if [v["step"] for v in checkpoint["candidates"]] != list(
            range(self.config.cadence, step + 1, self.config.cadence)
        ) or any(v["role"] != "original_development" for v in checkpoint["candidates"]):
            raise ValueError("Resume DEV opportunity sequence differs.")
        _strict_state(self.model, checkpoint["model"], assign=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        for k in ("samples", "candidates", "selected_step", "selected_state", "best_score"):
            setattr(self, k, copy.deepcopy(checkpoint[k]))
        self.initial_encoder, self.step, self.elapsed = (
            checkpoint["initial_encoder"],
            step,
            checkpoint["elapsed_seconds"],
        )
        core.restore_rng(checkpoint["rng"])

    def inference_artifact(self):
        if self.selected_state is None:
            raise ValueError("NOT_ASSESSABLE: no eligible original-DEV selected checkpoint.")
        return {
            "kind": "native_prefix_transfer_inference_v1",
            "config": self.config.to_dict(),
            "backbone_config": self.backbone,
            "architecture": ARCHITECTURES[self.config.family],
            "scalers": self.statistics,
            "model": self.selected_state,
            "selected_step": self.selected_step,
            "identities": self.identities,
            "feature_ancestor": copy.deepcopy(self.model.feature_ancestor),
            "training_kind": "supervised_prefix_readout_on_" + TRAINING_KINDS[self.config.method]
            if self.config.mode == "frozen_readout"
            else "scratch_supervised_prefix_encoder",
            "zero_shot": False,
        }


class PrefixPredictor:
    """Safe portable context-only forecast; loading never resets RNG or fits."""

    def __init__(self, weights, *, device="cpu"):
        artifact = torch.load(weights, weights_only=True, map_location="cpu")
        if artifact.get("kind") != "native_prefix_transfer_inference_v1":
            raise ValueError("Distinct owned prefix inference kind required.")
        self.config = PrefixConfig(**artifact["config"])
        self.config.validate(
            EVIDENCE if self.config.correctness_smoke else "REVIEWED_PREFIX_TRANSFER"
        )
        if (
            artifact.get("architecture") != ARCHITECTURES[self.config.family]
            or artifact.get("zero_shot") is not False
            or not artifact.get("identities")
        ):
            raise ValueError("Explicit prefix ancestry/architecture required.")
        if artifact.get("selected_step") not in range(
            self.config.cadence, self.config.updates + 1, self.config.cadence
        ):
            raise ValueError("Selected endpoint not a scheduled DEV opportunity.")
        self.scalers = _scalers(artifact["scalers"])
        _validate_backbone(self.config, artifact["backbone_config"])
        if device not in ("cpu", "cuda:0") or (self.config.correctness_smoke and device != "cpu"):
            raise ValueError("Synthetic replay CPU only; one declared local device.")
        self.device = device
        with torch.device("meta"):
            self.model = PrefixModel(
                artifact["backbone_config"], self.config.family, self.config.band_artifact_version
            ).float()
        _strict_state(self.model, artifact["model"])
        self.model.to(device).eval().requires_grad_(False)

    def forecast(self, x, observed, metadata, query):
        x, observed, metadata = _inputs(x, observed, metadata, 96)
        if not np.allclose(metadata[:, 0, 3:5] * 250, [0, 230], atol=1e-5, rtol=0):
            raise ValueError("Prefix-transfer suffix forecasting preserves native0–230m.")
        if (
            query.shape != (len(x), 3, 10)
            or not np.isfinite(query).all()
            or not np.allclose(query[..., :9], metadata[:, :1, :9])
            or not np.array_equal(query[..., 9], np.broadcast_to(HORIZONS, (len(x), 3)))
        ):
            raise ValueError("Exact issued native query required.")
        data = {"x": x, "context_observed": observed, "metadata": metadata, "query": query}
        config = PrefixConfig(**{**self.config.to_dict(), "device": self.device})
        prediction = predict(self.model, data, self.scalers, config)
        if not np.isfinite(prediction).all():
            raise ValueError("Finite forecasts required for every unchanged issuance.")
        return prediction


def fit(manifest_path, review_path, output_path, *, resume=None):
    """Owner-only approved execution. No suffix prediction, calibration or selection."""
    admission = admit(manifest_path, review_path, output_path, resume=resume)
    admission.manifest["_base"] = str(Path(manifest_path).resolve().parent)
    m, config, base = admission.manifest, admission.config, Path(admission.manifest["_base"])
    if (
        not admission.partition["fit_rows"]
        or admission.partition["suffix_support_status"] == "NOT_ASSESSABLE"
    ):
        return {
            "kind": "native_prefix_transfer_completion_v1",
            "evidence_kind": admission.manifest["evidence_kind"],
            "status": "NOT_ASSESSABLE",
            "reason": "fixed metadata prefix/suffix support insufficient",
            "partition": admission.partition,
        }
    output = Path(output_path).resolve()
    backbone = admission.documents[_path(m["backbone_config"], base)]
    statistics = admission.documents[_path(m["scalers"], base)]
    selected = (
        decode_checkpoint(admission.snapshots[_path(m["encoder"], base)])
        if config.mode == "frozen_readout"
        else None
    )
    parent = admission.documents[_path(m["parent_run"], base)] if selected is not None else None
    if selected is not None:
        expected_selected_kind = selected_kind(config)
        if (
            selected.get("kind") != expected_selected_kind
            or selected.get("config") != backbone
            or selected.get("scalers") != statistics
            or selected.get("bindings") != parent["bindings"]
        ):
            raise ValueError("Selected safe encoder kind/config/scalers/source lineage differs.")
        with torch.device("meta"):
            template = PrefixModel(backbone, config.family, config.band_artifact_version).float()
        _strict_state(template.encoder, selected.get("encoder"))
        parent_artifact = decode_checkpoint(admission.snapshots[_path(m["parent_inference"], base)])
        inference_kind = (
            (
                "native_band_replication_ssl_weights_only_inference_v2"
                if config.band_artifact_version == 2
                else "native_band_ssl_weights_only_inference_v1"
            )
            if config.family == "band"
            else "native_ssl_weights_only_inference_v1"
        )
        if (
            parent_artifact.get("kind") != inference_kind
            or parent_artifact.get("config") != backbone
            or parent_artifact.get("scalers") != statistics
            or parent_artifact.get("bindings") != parent["bindings"]
        ):
            raise ValueError("Safe selected parent inference lineage mismatch.")
        if config.method == "direct":
            lineage = _supervised_ancestry(parent.get("supervised_ancestry"), parent["config"])
            if (
                selected.get("supervised_ancestry") != lineage
                or parent_artifact.get("supervised_ancestry") != lineage
                or parent_artifact.get("downstream_config") != parent["config"]
                or parent_artifact.get("review_sha256") != parent["review_sha256"]
                or selected.get("evidence_kind") != parent["evidence_kind"]
                or parent_artifact.get("evidence_kind") != parent["evidence_kind"]
            ):
                raise ValueError("Safe direct artifact/config/supervised feature lineage differs.")
            if config.family == "band" and any(
                a.get("architecture") != ARCHITECTURES["band"] for a in (selected, parent_artifact)
            ):
                raise ValueError("Typed supervised band artifact architecture differs.")
            with torch.device("meta"):
                if config.family == "band":
                    from marine_echo.models.native_band_temporal import NativeBandTemporalModel

                    full_template = NativeBandTemporalModel(
                        **core.model_dimensions(
                            _backbone(backbone, config.family, config.band_artifact_version)
                        )
                    ).float()
                else:
                    full_template = core.native_temporal.NativeTemporalModel(
                        **core.model_dimensions(
                            _backbone(backbone, config.family, config.band_artifact_version)
                        )
                    ).float()
            _strict_state(full_template, parent_artifact.get("model"))
        parent_state = {
            k.removeprefix("encoder."): v
            for k, v in parent_artifact["model"].items()
            if k.startswith("encoder.")
        }
        if set(parent_state) != set(selected["encoder"]) or any(
            not torch.equal(v, selected["encoder"][k]) for k, v in parent_state.items()
        ):
            raise ValueError("Selected encoder differs from parent inference features.")
    prefix, dev = (
        _decode(admission, "prefix_npz", "prefix"),
        _decode(admission, "dev_npz", "development"),
    )
    if set(prefix["deployment"]) & set(dev["deployment"]):
        raise ValueError("Original DEV must be outside the adapted deployment.")
    if not len(prefix["x"]) or not prefix["target_observed"].any(0).all():
        return {
            "kind": "native_prefix_transfer_completion_v1",
            "evidence_kind": admission.manifest["evidence_kind"],
            "status": "NOT_ASSESSABLE",
            "reason": "no admissible prefix row per horizon",
            "partition": admission.partition,
        }
    if resume is None:
        output.mkdir(parents=False, exist_ok=False)
    with core.Resources(config.device, output) as resources:
        trajectory = _Trajectory(
            config,
            backbone,
            selected,
            statistics,
            prefix,
            dev,
            admission.identities,
            admission=admission,
            parent_bindings=parent["bindings"] if parent else None,
        )
        if resume is not None:
            trajectory.restore(decode_checkpoint(admission.snapshots[Path(resume).resolve()]))
        while trajectory.step < config.updates:
            trajectory.advance(
                min(config.updates, ((trajectory.step // config.cadence) + 1) * config.cadence)
            )
            resources.check()
            core.atomic_checkpoint(output / "latest.pt", trajectory.checkpoint())
        receipt = {
            "kind": "native_prefix_transfer_completion_v1",
            "evidence_kind": admission.manifest["evidence_kind"],
            "status": "COMPLETED" if trajectory.selected_state is not None else "NOT_ASSESSABLE",
            "config": config.to_dict(),
            "architecture": ARCHITECTURES[config.family],
            "bindings": admission.identities,
            "partition": admission.partition,
            "prefit_review_sha256": sha(Path(review_path).read_bytes()),
            "prefix_ancestors_explicit": True,
            "zero_shot": False,
            "suffix_numerical_access": False,
            "original_train_scalers": statistics,
            "feature_ancestor": copy.deepcopy(trajectory.model.feature_ancestor),
            "parent_run_sha256": admission.identities.get(str(_path(m["parent_run"], base)))
            if parent is not None
            else None,
            "parent_inference_sha256": admission.identities.get(
                str(_path(m["parent_inference"], base))
            )
            if parent is not None
            else None,
            "external_dev_exposure": "original_DEV0–225_selection_only",
            "label_counts": prefix["target_observed"].sum(0).tolist(),
            "unique_prefix_rows": len(prefix["x"]),
            "support_sha256": sha(prefix["target_observed"].tobytes()),
            "observed_prefix_target_values_sha256": sha(
                np.where(prefix["target_observed"], prefix["targets"], 0)
                .astype(np.float32)
                .tobytes()
            ),
            "sample_sequence_sha256": sha(json.dumps(trajectory.samples).encode()),
            "updates": trajectory.step,
            "candidates": trajectory.candidates,
            "selected_step": trajectory.selected_step,
            "encoder_parameters": sum(p.numel() for p in trajectory.model.encoder.parameters()),
            "head_parameters": sum(p.numel() for p in trajectory.model.head.parameters()),
            "parameter_matching": "CF128/head18565 versus shared64/head5189; not cross-architecture matched",
            "encoder_unchanged": all(
                torch.equal(v, trajectory.model.encoder.state_dict()[k].cpu())
                for k, v in trajectory.initial_encoder.items()
            ),
            "resources": resources.snapshot(),
            "elapsed_trajectory_seconds": trajectory.elapsed,
            "suffix_status": "NOT_RUN_requires_separate_reviewed_inference_and_floor18_support",
            "scientific_claim": None,
        }
        if config.mode == "scratch_direct" and receipt["encoder_unchanged"]:
            raise RuntimeError("Scratch supervised trajectory made no encoder changes.")
        if trajectory.selected_state is not None:
            with (output / "inference.pt").open("xb") as stream:
                stream.write(encode_checkpoint(trajectory.inference_artifact()))
        with (output / "completion.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {"status": fit(args.manifest, args.review, args.output, resume=args.resume)["status"]}
        )
    )


if __name__ == "__main__":
    main()
