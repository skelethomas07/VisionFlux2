from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BatchInput:
    item_id: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class BatchProgress:
    overall_fraction: float
    file_fraction: float
    file_index: int
    total_files: int
    filename: str
    message: str
    elapsed_seconds: float


@dataclass
class BatchOutcome:
    item_id: str
    filename: str
    result: Any | None
    error: str | None
    duration_seconds: float


def format_elapsed(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분 {secs:02d}초"
    return f"{minutes}분 {secs:02d}초"


def run_batch(
    inputs: Iterable[BatchInput],
    analyze: Callable[[BatchInput, Callable[[float, str], None]], Any],
    *,
    on_progress: Callable[[BatchProgress], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[BatchOutcome]:
    items = list(inputs)
    total = len(items)
    if total == 0:
        return []

    started = clock()
    outcomes: list[BatchOutcome] = []

    def emit(
        overall: float,
        file_fraction: float,
        file_index: int,
        filename: str,
        message: str,
    ) -> None:
        if on_progress is None:
            return
        on_progress(
            BatchProgress(
                overall_fraction=float(min(1.0, max(0.0, overall))),
                file_fraction=float(min(1.0, max(0.0, file_fraction))),
                file_index=int(file_index),
                total_files=total,
                filename=filename,
                message=str(message),
                elapsed_seconds=max(0.0, float(clock() - started)),
            )
        )

    emit(0.0, 0.0, 0, items[0].filename, "분석 준비")

    for zero_index, item in enumerate(items):
        file_started = clock()
        last_fraction = 0.0

        def report(file_fraction: float, message: str) -> None:
            nonlocal last_fraction
            last_fraction = max(last_fraction, min(1.0, max(0.0, float(file_fraction))))
            overall = (zero_index + last_fraction) / total
            emit(overall, last_fraction, zero_index + 1, item.filename, message)

        report(0.0, "파일 분석 시작")
        result = None
        error = None
        try:
            result = analyze(item, report)
            report(1.0, "파일 분석 완료")
        except Exception as exc:  # Per-file isolation is intentional.
            error = f"{type(exc).__name__}: {exc}"
            report(1.0, f"파일 분석 실패: {error}")

        outcomes.append(
            BatchOutcome(
                item_id=item.item_id,
                filename=item.filename,
                result=result,
                error=error,
                duration_seconds=max(0.0, float(clock() - file_started)),
            )
        )

    emit(1.0, 1.0, total, items[-1].filename, "전체 분석 완료")
    return outcomes
