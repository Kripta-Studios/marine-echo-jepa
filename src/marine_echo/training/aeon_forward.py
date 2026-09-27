"""TRAIN-only six-future-product pairing for the forward AEON study."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marine_echo.training.aeon_corpus import AeonDevelopmentReader, AeonHourlySlot
from marine_echo.training.aeon_development import _sha256
from marine_echo.training.aeon_windows import (
    AeonHourlyWindow, AeonWindowPlan, _roughly_hourly, iter_aeon_windows,
)


@dataclass(frozen=True)
class ForwardPairs:
    """Exact TRAIN subset for six-product self-supervision."""

    fit_indices: np.ndarray
    row_ids: tuple[str, ...]
    cutoff_interval_ids: np.ndarray
    future_interval_ids: np.ndarray
    future_db: np.ndarray
    future_mask: np.ndarray
    source_archive_sha256: str


def build_forward_pairs(
    fit: list[AeonHourlyWindow], slots: list[AeonHourlySlot]
) -> ForwardPairs:
    """Use future observations only as TRAIN pretraining targets."""
    if not fit or not slots or any(row.partition != "train" for row in fit):
        raise ValueError("Forward pairs require only TRAIN issue rows and source slots.")
    source_hashes = {row.source_archive_sha256 for row in fit} | {
        slot.archive_sha256 for slot in slots
    }
    if len(source_hashes) != 1:
        raise ValueError("Forward pairs mix source archives.")
    by_id = {slot.interval_id: slot for slot in slots}
    if len(by_id) != len(slots):
        raise ValueError("Forward TRAIN source intervals are duplicated.")
    selected: list[int] = []
    future_values = []
    future_masks = []
    future_ids = []
    for fit_index, row in enumerate(fit):
        ids = row.cutoff_interval_id + np.arange(1, 7)
        future = [by_id.get(int(interval_id)) for interval_id in ids]
        if any(item is None for item in future):
            continue
        complete = [item for item in future if item is not None]
        if (
            any(not item.observed_mask[0] for item in complete)
            or any(not _roughly_hourly(row.cutoff_source_timestamp, item.source_timestamp, offset)
                   for offset, item in enumerate(complete, start=1))
            or not np.array_equal(row.target_interval_ids, ids[[0, 2, 5]])
            or any(item.source_timestamp >= np.datetime64("2024-10-08", "us")
                   for item in complete)
        ):
            continue
        selected.append(fit_index)
        future_values.append(np.stack([item.sv_db for item in complete]))
        future_masks.append(np.stack([item.observed_mask for item in complete]))
        future_ids.append(ids)
    if not selected:
        raise ValueError("No exact six-product forward pairs remain in TRAIN.")
    return ForwardPairs(
        fit_indices=np.asarray(selected, dtype=np.int64),
        row_ids=tuple(fit[index].row_id for index in selected),
        cutoff_interval_ids=np.asarray([fit[index].cutoff_interval_id for index in selected]),
        future_interval_ids=np.stack(future_ids),
        future_db=np.stack(future_values),
        future_mask=np.stack(future_masks),
        source_archive_sha256=source_hashes.pop(),
    )


def load_train_forward_cohort(
    archive: Path, split_review: Path
) -> tuple[list[AeonHourlyWindow], ForwardPairs, str]:
    """Open the approved TRAIN date rows only; never read validation outcomes."""
    split_sha256 = _sha256(split_review)
    reader = AeonDevelopmentReader(
        archive, review_path=split_review, review_sha256=split_sha256
    )
    slots = list(reader.iter_partition("train"))
    fit = list(iter_aeon_windows(
        slots, plan=AeonWindowPlan(), partition="train",
        partition_start="2024-03-06", partition_end_exclusive="2024-10-08",
    ))
    return fit, build_forward_pairs(fit, slots), split_sha256


def pair_inventory(fit: list[AeonHourlyWindow], pairs: ForwardPairs) -> dict[str, object]:
    if not fit or not pairs.row_ids:
        raise ValueError("Forward pair inventory is empty.")
    digest = hashlib.sha256()
    for row_id, ids in zip(pairs.row_ids, pairs.future_interval_ids, strict=True):
        digest.update(row_id.encode("ascii"))
        digest.update(np.asarray(ids, dtype=np.int64).tobytes())
    value_digest = hashlib.sha256()
    value_digest.update(np.ascontiguousarray(pairs.future_db).tobytes())
    value_digest.update(np.ascontiguousarray(pairs.future_mask).tobytes())
    return {
        "classification": "TRAIN_ONLY_PREFIT_PAIR_INVENTORY_NOT_EVALUATION",
        "source_archive_sha256": pairs.source_archive_sha256,
        "train_issued_rows": len(fit),
        "six_future_pair_rows": len(pairs.row_ids),
        "pair_fraction_of_train_issue_rows": len(pairs.row_ids) / len(fit),
        "first_cutoff_interval_id": int(pairs.cutoff_interval_ids[0]),
        "last_cutoff_interval_id": int(pairs.cutoff_interval_ids[-1]),
        "pair_id_and_interval_sha256": digest.hexdigest(),
        "pair_target_values_and_mask_sha256": value_digest.hexdigest(),
        "future_interval_ids": pairs.future_interval_ids.astype(int).tolist(),
        "row_ids": list(pairs.row_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit, pairs, split_sha256 = load_train_forward_cohort(args.archive, args.split_review)
    result = pair_inventory(fit, pairs)
    result["split_review_sha256"] = split_sha256
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Forward pair inventory already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in (
        "future_interval_ids", "row_ids"
    )}, indent=2))


if __name__ == "__main__":
    main()
