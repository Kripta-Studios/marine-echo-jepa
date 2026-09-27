"""TRAIN-only future-product pairing for forward EMA."""

from __future__ import annotations

import numpy as np

from marine_echo.training.aeon_forward import build_forward_pairs
from marine_echo.training.aeon_corpus import AeonHourlySlot
from marine_echo.training.aeon_windows import AeonWindowPlan, iter_aeon_windows


def _slots() -> list[AeonHourlySlot]:
    result = []
    first = np.datetime64("2024-03-06T00:52:00", "us")
    for index in range(40):
        result.append(AeonHourlySlot(
            interval_id=500000 + index,
            source_timestamp=first + index * np.timedelta64(1, "h"),
            sv_db=np.full(4, -80.0 + index / 100),
            observed_mask=np.ones(4, dtype=bool),
            qc_status=("OBSERVED_SOURCE_PRODUCT",) * 4,
            member_names=("train.csv",), archive_sha256="a" * 64,
        ))
    return result


def test_forward_pairs_use_only_exact_future_train_products() -> None:
    slots = _slots()
    fit = list(iter_aeon_windows(
        slots, plan=AeonWindowPlan(), partition="train",
        partition_start="2024-03-06", partition_end_exclusive="2024-03-09",
    ))
    pairs = build_forward_pairs(fit, slots)
    assert len(pairs.row_ids) == len(fit)
    assert pairs.future_db.shape == (len(fit), 6, 4)
    assert pairs.future_mask.all()
    assert np.array_equal(pairs.future_interval_ids[0], np.arange(500024, 500030))
    assert pairs.row_ids[0] == fit[0].row_id

    broken = list(slots)
    slot = broken[24]
    mask = slot.observed_mask.copy()
    mask[0] = False
    from dataclasses import replace
    broken[24] = replace(slot, observed_mask=mask,
                         qc_status=("INVALID_OR_SPECIAL_SV",) + slot.qc_status[1:])
    reduced = build_forward_pairs(fit, broken)
    assert fit[0].row_id not in reduced.row_ids
    assert len(reduced.row_ids) < len(pairs.row_ids)
