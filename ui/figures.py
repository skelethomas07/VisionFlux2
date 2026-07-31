from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_STATUS_STYLE = {
    "active": ("#37d67a", 1.35),
    "accepted": ("#ffd84d", 1.8),
    "rejected": ("#ff4d5a", 1.0),
    "corrected": ("#7a7a7a", 1.0),
}


def _rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3]


def build_overlay_figure(
    image: np.ndarray,
    measurements: pd.DataFrame,
    selected_measurement_id: str | None = None,
    show_rejected: bool = False,
) -> go.Figure:
    h, w = image.shape[:2]
    fig = go.Figure()
    fig.add_trace(go.Image(z=_rgb(image), hoverinfo="skip", name="SEM"))

    if measurements is None or measurements.empty:
        visible = pd.DataFrame()
    else:
        visible = measurements.copy()
        if not show_rejected:
            visible = visible[visible["status"].astype(str).isin(["active", "accepted"])]

    for status, group in visible.groupby("status", sort=False) if not visible.empty else []:
        color, width = _STATUS_STYLE.get(str(status), ("#37d67a", 1.3))
        if "source" in group.columns and (group["source"].astype(str) == "manual").all():
            color, width = "#00e5ff", 2.4
        xs: list[float | None] = []
        ys: list[float | None] = []
        for _, row in group.iterrows():
            xs.extend([row.get("x1"), row.get("x2"), None])
            ys.extend([row.get("y1"), row.get("y2"), None])
        fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": width},
            hoverinfo="skip",
            name=f"{status} thickness",
            showlegend=True,
        ))

    if not visible.empty:
        mids_x = pd.to_numeric(visible["center_x"], errors="coerce")
        mids_y = pd.to_numeric(visible["center_y"], errors="coerce")
        width_original = pd.to_numeric(
            visible.get("width_original_px", visible.get("width_px")), errors="coerce"
        )
        nm_source = visible["width_nm"] if "width_nm" in visible.columns else pd.Series(np.nan, index=visible.index)
        nm_values = pd.to_numeric(nm_source, errors="coerce")
        selectedpoints = []
        if selected_measurement_id is not None:
            selectedpoints = np.flatnonzero(
                visible["measurement_id"].astype(str).to_numpy() == str(selected_measurement_id)
            ).tolist()
        region_values = visible["fiber_region_id"].astype(str) if "fiber_region_id" in visible.columns else pd.Series([""] * len(visible), index=visible.index)
        source_values = visible["source"].astype(str) if "source" in visible.columns else pd.Series([""] * len(visible), index=visible.index)
        custom = [
            [str(mid), str(region), float(width), float(nm) if np.isfinite(nm) else np.nan, str(source)]
            for mid, region, width, nm, source in zip(
                visible["measurement_id"], region_values, width_original, nm_values, source_values
            )
        ]
        fig.add_trace(go.Scattergl(
            x=mids_x,
            y=mids_y,
            mode="markers",
            name="click a measurement",
            customdata=custom,
            marker={
                "size": 7,
                "color": "rgba(255,255,255,0.24)",
                "line": {"color": "rgba(0,0,0,0.55)", "width": 0.6},
            },
            selectedpoints=selectedpoints,
            selected={"marker": {"color": "#ff2bd6", "size": 13, "opacity": 1}},
            unselected={"marker": {"opacity": 0.45}},
            hovertemplate=(
                "ID=%{customdata[0]}<br>region=%{customdata[1]}"
                "<br>width=%{customdata[2]:.2f} original px"
                "<br>width_nm=%{customdata[3]:.2f}<br>source=%{customdata[4]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        title="Automatic thickness overlay — click a midpoint marker to review",
        height=min(900, max(520, int(720 * h / max(w, 1)))),
        dragmode="pan",
        clickmode="event+select",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
    )
    fig.update_xaxes(range=[0, w], visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x", scaleratio=1)
    return fig


def build_distribution_figure(
    representatives: pd.DataFrame,
    use_nm: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if representatives is None or representatives.empty:
        fig.update_layout(title="표시할 대표 두께가 없습니다")
        return fig
    value_col = "representative_width_nm" if use_nm else "representative_width_original_px"
    if value_col not in representatives.columns:
        value_col = "representative_width_px"
        use_nm = False
    values = pd.to_numeric(representatives[value_col], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(representatives["fiber_count_weight"], errors="coerce").fillna(0).to_numpy(float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values):
        fig.update_layout(title="표시할 대표 두께가 없습니다")
        return fig
    bins = min(30, max(6, int(np.sqrt(len(values)) * 2)))
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges), name="fiber 영역 가중치"))
    order = np.argsort(values)
    csum = np.cumsum(weights[order])
    weighted_median = values[order][np.searchsorted(csum, 0.5 * csum[-1])]
    fig.add_vline(x=float(weighted_median), line_dash="dash", annotation_text=f"중앙값 {weighted_median:.2f}")
    fig.update_layout(
        title="Fiber 영역별 대표 두께 분포",
        xaxis_title="두께 (nm)" if use_nm else "두께 (원본 px)",
        yaxis_title="fiber 영역 가중치",
        bargap=0.05,
        height=430,
    )
    return fig


def _orientation_color(angle_deg: float) -> str:
    import colorsys

    hue = (float(angle_deg) % 180.0) / 180.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def build_orientation_histogram(orientation) -> go.Figure:
    centers = np.asarray(orientation.histogram_centers_deg, dtype=float)
    values = np.asarray(orientation.histogram_fraction, dtype=float)
    colors = [_orientation_color(angle) for angle in centers]
    fig = go.Figure(go.Bar(
        x=centers,
        y=values,
        marker_color=colors,
        hovertemplate="방향 %{x:.1f}°<br>가중 비율 %{y:.4f}<extra></extra>",
    ))
    if np.isfinite(orientation.dominant_direction_deg):
        fig.add_vline(
            x=float(orientation.dominant_direction_deg),
            line_dash="dash",
            annotation_text=f"주방향 {orientation.dominant_direction_deg:+.1f}°",
        )
    fig.update_layout(
        title="방향 분포",
        xaxis_title="fiber 방향 (°)",
        yaxis_title="coherency 가중 비율",
        xaxis={"range": [-90, 90], "tickvals": [-90, -45, 0, 45, 90]},
        height=390,
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
        showlegend=False,
    )
    return fig


def build_orientation_rose(orientation, bins: int = 36) -> go.Figure:
    theta = np.asarray(orientation.theta, dtype=float)
    weights = np.asarray(orientation.coherency, dtype=float)
    gate = np.asarray(orientation.gate, dtype=bool)
    edges = np.linspace(-90.0, 90.0, int(bins) + 1)
    hist, _ = np.histogram(theta[gate], bins=edges, weights=weights[gate])
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    theta_full = np.concatenate([centers % 360.0, (centers + 180.0) % 360.0])
    r_full = np.concatenate([hist, hist])
    colors = [_orientation_color(angle) for angle in np.concatenate([centers, centers])]
    fig = go.Figure(go.Barpolar(
        r=r_full,
        theta=theta_full,
        width=np.full_like(theta_full, 180.0 / bins),
        marker_color=colors,
        marker_line_color="rgba(255,255,255,.35)",
        marker_line_width=0.4,
        hovertemplate="방향 %{theta:.1f}°<br>가중 비율 %{r:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"방향 장미도 · 정렬도 S={orientation.order_parameter:.3f}",
        polar={
            "angularaxis": {"direction": "counterclockwise", "rotation": 0},
            "radialaxis": {"showticklabels": False, "ticks": ""},
        },
        height=390,
        margin={"l": 35, "r": 35, "t": 55, "b": 25},
        showlegend=False,
    )
    return fig


def build_fiber_direction_figure(representative_lines: list[dict]) -> go.Figure:
    directions = []
    weights = []
    for line in representative_lines:
        value = line.get("direction_deg")
        if value is None:
            continue
        try:
            angle = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(angle):
            directions.append(angle)
            weights.append(1.0)
    fig = go.Figure()
    if not directions:
        fig.update_layout(title="대표 fiber 방향 데이터가 없습니다", height=360)
        return fig
    hist, edges = np.histogram(directions, bins=36, range=(-90, 90), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(go.Bar(
        x=centers,
        y=hist,
        marker_color=[_orientation_color(a) for a in centers],
        hovertemplate="방향 %{x:.1f}°<br>대표 fiber 수 %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="대표 fiber 기준 방향 분포",
        xaxis_title="fiber 방향 (°)",
        yaxis_title="대표 fiber 수",
        xaxis={"range": [-90, 90]},
        height=360,
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
        showlegend=False,
    )
    return fig


def build_direction_segment_figure(measurements: pd.DataFrame, bins: int = 36) -> go.Figure:
    """Length-weighted direction distribution from local fiber path segments.

    A long curved fiber contributes its local directions along the path instead of
    being forced into one representative angle. Rejected/corrected rows are excluded.
    """
    fig = go.Figure()
    if measurements is None or measurements.empty or "direction_deg" not in measurements.columns:
        fig.update_layout(title="Fiber 방향 구간 데이터가 없습니다", height=360)
        return fig
    active = measurements.copy()
    if "status" in active.columns:
        active = active[active["status"].astype(str).isin(["active", "accepted"])]
    directions = pd.to_numeric(active["direction_deg"], errors="coerce").to_numpy(float)
    if "sample_length_px" in active.columns:
        weights = pd.to_numeric(active["sample_length_px"], errors="coerce").fillna(0).to_numpy(float)
    else:
        weights = np.ones(len(active), dtype=float)
    good = np.isfinite(directions) & np.isfinite(weights) & (weights > 0)
    directions, weights = directions[good], weights[good]
    if not len(directions):
        fig.update_layout(title="Fiber 방향 구간 데이터가 없습니다", height=360)
        return fig
    hist, edges = np.histogram(directions, bins=int(bins), range=(-90, 90), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(go.Bar(
        x=centers,
        y=hist,
        marker_color=[_orientation_color(a) for a in centers],
        hovertemplate="방향 %{x:.1f}°<br>구간 길이 %{y:.1f} px<extra></extra>",
    ))
    fig.update_layout(
        title="Fiber 국소 방향 분포",
        xaxis_title="fiber 방향 (°)",
        yaxis_title="검출된 구간 길이 (px)",
        xaxis={"range": [-90, 90], "tickvals": [-90, -45, 0, 45, 90]},
        height=360,
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
        showlegend=False,
    )
    return fig
