from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pipeline.review import recompute_representatives


@dataclass
class ReviewItem:
    item_id: str
    analysis: Any
    measurements: pd.DataFrame
    representatives: pd.DataFrame
    feedback: list[dict] = field(default_factory=list)
    revision: int = 0
    nm_per_px: float | None = None
    last_apply_token: str | None = None
    duration_seconds: float = 0.0


def build_review_item(
    item_id: str,
    analysis: Any,
    *,
    nm_per_px: float | None,
    duration_seconds: float = 0.0,
) -> ReviewItem:
    item = ReviewItem(
        item_id=str(item_id),
        analysis=analysis,
        measurements=analysis.measurements.copy(deep=True),
        representatives=pd.DataFrame(),
        feedback=[],
        revision=0,
        nm_per_px=nm_per_px,
        last_apply_token=None,
        duration_seconds=float(duration_seconds),
    )
    recompute_review_item(item)
    return item


def recompute_review_item(item: ReviewItem) -> None:
    item.representatives = recompute_representatives(
        item.measurements,
        analysis_scale=float(item.analysis.analysis_scale),
        nm_per_px=item.nm_per_px,
    )
