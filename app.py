from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.analyzer import AnalysisResult, run_uploaded_analysis
from pipeline.review import (
    apply_canvas_edits,
    build_representative_lines,
    build_session_zip,
    recompute_representatives,
)
from ui.figures import (
    build_distribution_figure,
    build_fiber_direction_figure,
    build_orientation_histogram,
    build_orientation_rose,
)
from ui.measurement_canvas import measurement_canvas, normalize_canvas_payload

SUPPORTED_TYPES = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]


@st.cache_data(max_entries=3, show_spinner=False)
def _cached_analysis(
    image_bytes: bytes,
    filename: str,
    nm_per_px: float | None,
    max_dimension: int | None,
) -> AnalysisResult:
    return run_uploaded_analysis(
        image_bytes,
        filename,
        nm_per_px=nm_per_px,
        max_dimension=max_dimension,
    )


def _init_state() -> None:
    defaults = {
        "analysis": None,
        "measurements": pd.DataFrame(),
        "representatives": pd.DataFrame(),
        "feedback": [],
        "revision": 0,
        "upload_digest": None,
        "nm_per_px": None,
        "last_apply_token": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _recompute() -> None:
    analysis = st.session_state.analysis
    if analysis is None:
        st.session_state.representatives = pd.DataFrame()
        return
    st.session_state.representatives = recompute_representatives(
        st.session_state.measurements,
        analysis_scale=analysis.analysis_scale,
        nm_per_px=st.session_state.nm_per_px,
    )


def _run_analysis(uploaded, nm_per_px: float | None, max_dimension: int | None) -> None:
    image_bytes = uploaded.getvalue()
    digest = hashlib.sha256(image_bytes).hexdigest()
    with st.spinner("두께와 방향을 분석하고 있습니다. 이미지 크기에 따라 잠시 걸릴 수 있습니다."):
        result = _cached_analysis(image_bytes, uploaded.name, nm_per_px, max_dimension)
    st.session_state.analysis = result
    st.session_state.measurements = result.measurements.copy(deep=True)
    st.session_state.feedback = []
    st.session_state.revision += 1
    st.session_state.upload_digest = digest
    st.session_state.nm_per_px = nm_per_px
    st.session_state.last_apply_token = None
    _recompute()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return float("nan")
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _sidebar() -> None:
    st.sidebar.markdown("## 분석")
    uploaded = st.sidebar.file_uploader(
        "SEM 이미지",
        type=SUPPORTED_TYPES,
        help="TIFF, PNG, JPEG, BMP 파일 한 장을 올립니다.",
    )

    use_calibration = st.sidebar.toggle(
        "nm 단위 사용",
        value=False,
        help="이미지의 원본 픽셀 크기(nm/px)를 알고 있을 때 켭니다.",
    )
    nm_per_px = None
    if use_calibration:
        nm_per_px = st.sidebar.number_input(
            "원본 이미지 nm/px",
            min_value=0.000001,
            value=1.0,
            format="%.6f",
            help="스케일 바 또는 촬영 조건에서 확인한 원본 이미지의 nm/px 값입니다.",
        )

    with st.sidebar.expander("분석 해상도", expanded=False):
        resolution = st.radio(
            "처리 크기",
            ["빠름 · 최대 1200 px", "균형 · 최대 1600 px", "원본 · 로컬 권장"],
            index=0,
            help="Cloud에서는 1200 px가 가장 안정적입니다. 결과 두께는 원본 픽셀 크기로 환산됩니다.",
        )
    max_dimension = {
        "빠름 · 최대 1200 px": 1200,
        "균형 · 최대 1600 px": 1600,
        "원본 · 로컬 권장": None,
    }[resolution]

    if st.sidebar.button(
        "분석 시작",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
        help="Edge, ridge, OrientationJ 방향 정보와 원본 SEM 일치도를 함께 사용해 두께를 찾습니다.",
    ):
        try:
            _run_analysis(uploaded, nm_per_px, max_dimension)
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"분석 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")

    if st.session_state.analysis is not None:
        st.sidebar.divider()
        st.sidebar.caption(
            "노란 선은 자동 대표 두께, 하늘색 선은 방향 기반 보완, 청록 선은 사용자가 추가한 두께입니다."
        )


def _summary_metrics() -> None:
    reps = st.session_state.representatives
    measurements = st.session_state.measurements
    use_nm = st.session_state.nm_per_px is not None
    value_col = "representative_width_nm" if use_nm else "representative_width_original_px"
    values = pd.to_numeric(reps.get(value_col, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    weights = pd.to_numeric(reps.get("fiber_count_weight", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(float)
    median = _weighted_median(values, weights) if len(values) else float("nan")
    manual_count = int((measurements.get("source", pd.Series(dtype=str)).astype(str) == "manual").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("대표 두께 수", int(len(reps)), help="같은 fiber 영역에서 반복 측정된 값은 하나의 대표값으로 묶습니다.")
    c2.metric(
        "중앙 두께",
        "—" if not np.isfinite(median) else f"{median:.2f} {'nm' if use_nm else 'px'}",
        help="각 fiber 영역이 같은 총 가중치를 갖도록 계산한 중앙값입니다.",
    )
    c3.metric("수동 추가", manual_count, help="사용자가 두 점 클릭으로 추가해 최종 반영한 두께선 수입니다.")


def _handle_canvas_result(result) -> None:
    raw_payload = getattr(result, "apply", None) if result is not None else None
    if raw_payload is None:
        return
    payload = normalize_canvas_payload(raw_payload)
    if not payload["new_measurements"] and not payload["delete_ids"]:
        return
    token = f"{st.session_state.revision}:" + json.dumps(payload, sort_keys=True)
    if token == st.session_state.last_apply_token:
        return
    updated, events = apply_canvas_edits(
        st.session_state.measurements,
        payload["new_measurements"],
        payload["delete_ids"],
        analysis_scale=st.session_state.analysis.analysis_scale,
        nm_per_px=st.session_state.nm_per_px,
    )
    st.session_state.measurements = updated
    st.session_state.feedback.extend(events)
    st.session_state.last_apply_token = token
    st.session_state.revision += 1
    _recompute()
    st.toast(
        f"수동 측정 {len(payload['new_measurements'])}개 추가 · 기존 표시 {len(payload['delete_ids'])}개 수정"
    )
    st.rerun()


def _thickness_tab() -> None:
    analysis = st.session_state.analysis
    reps = st.session_state.representatives
    representative_lines = build_representative_lines(st.session_state.measurements, reps)

    with st.expander("측정값 고치는 방법", expanded=False):
        st.markdown(
            """
1. 마우스 휠로 확대하고 **이동** 도구로 원하는 위치를 찾습니다.  
2. **두께 추가**에서 fiber의 양쪽 edge를 차례로 클릭합니다. 여러 곳을 계속 표시할 수 있습니다.  
3. 잘못된 선은 **지우개**로 선택한 뒤 **전체 반영**을 누릅니다.
            """
        )

    canvas_result = measurement_canvas(
        analysis.image,
        representative_lines,
        analysis_scale=analysis.analysis_scale,
        nm_per_px=st.session_state.nm_per_px,
        revision=st.session_state.revision,
        key=f"visionflux-canvas-{analysis.image_name}",
    )
    _handle_canvas_result(canvas_result)

    st.markdown("### 두께 분포")
    use_nm = st.session_state.nm_per_px is not None
    st.plotly_chart(
        build_distribution_figure(reps, use_nm=use_nm),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    stem = Path(analysis.image_name).stem
    zip_bytes = build_session_zip(
        analysis.image_name,
        st.session_state.measurements,
        reps,
        st.session_state.feedback,
        analysis_summary=analysis.summary,
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "두께 분포 CSV",
        reps.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}_fiber_thickness_distribution.csv",
        mime="text/csv",
        use_container_width=True,
        help="3D 구조 생성에 사용할 fiber 영역별 대표 두께와 가중치를 저장합니다.",
    )
    d2.download_button(
        "검토 결과 ZIP",
        zip_bytes,
        file_name=f"{stem}_visionflux_review.zip",
        mime="application/zip",
        use_container_width=True,
        help="수정된 전체 측정, 대표 두께, 사용자 수정 기록과 분석 요약을 함께 저장합니다.",
    )


def _orientation_tab() -> None:
    analysis = st.session_state.analysis
    orientation = analysis.orientation
    if orientation is None:
        st.info("방향 결과가 없습니다. 이미지를 다시 분석해 주세요.")
        return
    representative_lines = build_representative_lines(
        st.session_state.measurements,
        st.session_state.representatives,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "주방향",
        "—" if not np.isfinite(orientation.dominant_direction_deg) else f"{orientation.dominant_direction_deg:+.1f}°",
        help="Coherency와 energy 조건을 통과한 구조 tensor를 모두 합쳐 계산한 방향입니다.",
    )
    m2.metric(
        "정렬도 S",
        "—" if not np.isfinite(orientation.order_parameter) else f"{orientation.order_parameter:.3f}",
        help="0은 방향이 고르게 퍼진 상태, 1은 거의 한 방향으로 정렬된 상태입니다.",
    )
    m3.metric(
        "방향 분석 면적",
        f"{100 * orientation.gated_fraction:.1f}%",
        help="방향 신뢰도와 구조 에너지 기준을 통과해 통계에 사용된 픽셀 비율입니다.",
    )

    st.caption("방향 기준: 0°는 수평, 양수는 화면에서 오른쪽으로 갈수록 위로 올라가는 방향입니다. Fiber 방향은 180° 주기입니다.")
    st.image(
        orientation.color_map,
        caption="색상은 방향, 채도는 방향 신뢰도(coherency), 밝기는 원본 SEM 명암을 나타냅니다.",
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_orientation_histogram(orientation), use_container_width=True, config={"displayModeBar": False})
    c2.plotly_chart(build_orientation_rose(orientation), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(
        build_fiber_direction_figure(representative_lines),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def main() -> None:
    st.set_page_config(
        page_title="VisionFlux — Surface to Volume",
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    st.title("VisionFlux")
    st.markdown("#### Surface to Volume")
    st.caption("SEM 한 장에서 fiber의 두께와 방향을 정리해, 3D 구조 생성에 사용할 입력 분포를 만듭니다.")

    _sidebar()
    if st.session_state.analysis is None:
        st.info("왼쪽에서 SEM 이미지를 올린 뒤 **분석 시작**을 눌러 주세요.")
        return

    _summary_metrics()
    thickness_tab, orientation_tab = st.tabs(["두께", "방향"])
    with thickness_tab:
        _thickness_tab()
    with orientation_tab:
        _orientation_tab()


if __name__ == "__main__":
    main()
