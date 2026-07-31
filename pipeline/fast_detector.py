from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import (
    closing,
    opening,
    disk,
    remove_small_objects,
    skeletonize,
)

from pipeline.compute import compute_multiscale_bright_ridge
from pipeline.orientation import OrientationResult, analyze_orientation, axial_error_deg, wrap90


@dataclass(frozen=True)
class FastDetectorConfig:
    ridge_sigmas: tuple[float, ...] = (1.0, 1.6, 2.4, 3.6, 5.2, 7.2)
    orientation_sigma_px: float = 3.2
    sample_spacing_px: float = 7.0
    min_path_length_px: float = 18.0
    min_component_pixels: int = 12
    max_half_width_px: float = 32.0
    min_width_px: float = 2.5
    max_width_px: float = 58.0
    ridge_percentile: float = 70.0
    high_ridge_percentile: float = 88.0
    min_coherency: float = 0.07
    direction_split_deg: float = 12.0
    direction_split_persistence: int = 3
    max_measurements: int = 5000
    min_confidence: float = 0.28
    pore_reject_fraction: float = 0.12


@dataclass
class FastDetectionResult:
    measurements: pd.DataFrame
    regions: pd.DataFrame
    representatives: pd.DataFrame
    candidates: pd.DataFrame
    orientation: OrientationResult
    pore_core: np.ndarray
    ridge_response: np.ndarray
    ridge_scale: np.ndarray
    summary: dict


def _emit(callback: Callable[[float, str], None] | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), str(message))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0).astype(np.float32)


def _robust_unit(field: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    positive = arr[arr > 0]
    if not positive.size:
        return np.zeros_like(arr, dtype=np.float32)
    scale = float(np.percentile(positive, percentile))
    return np.clip(arr / max(scale, 1e-9), 0.0, 1.0).astype(np.float32)


def build_pore_core(
    image: np.ndarray,
    energy: np.ndarray,
    ridge_response: np.ndarray,
) -> np.ndarray:
    """Build a conservative dark, low-energy pore interior mask.

    Coherency is deliberately not used as a hard pore criterion because crossings,
    curves, flat fiber interiors, and background can all have low coherency.
    """
    img = _normalize_image(image)
    local = ndimage.gaussian_filter(img, 5.0, mode="reflect")
    dark_depth = np.maximum(local - img, 0.0)
    energy_u = _robust_unit(energy, 99.0)
    ridge_u = _robust_unit(ridge_response, 99.5)

    positive_depth = dark_depth[dark_depth > 0]
    depth_thr = float(np.percentile(positive_depth, 58.0)) if positive_depth.size else 1.0
    dark_thr = float(np.percentile(img, 34.0))
    energy_thr = float(np.percentile(energy_u, 58.0))
    ridge_thr = float(np.percentile(ridge_u, 62.0))
    deep_thr = float(np.percentile(positive_depth, 84.0)) if positive_depth.size else 1.0

    low_structure = energy_u <= energy_thr
    enclosed_dark_gap = (dark_depth >= max(deep_thr, 0.05)) & (ridge_u <= max(ridge_thr, 0.16))
    pore = (
        (img <= dark_thr)
        & (dark_depth >= max(depth_thr, 0.015))
        & (low_structure | enclosed_dark_gap)
        & (ridge_u <= max(ridge_thr, 0.16))
    )
    pore = opening(pore, disk(1))
    pore = closing(pore, disk(1))
    try:
        pore = remove_small_objects(pore, max_size=7)
    except TypeError:
        pore = remove_small_objects(pore, min_size=8)
    return np.asarray(pore, dtype=bool)


def _candidate_skeleton(
    image: np.ndarray,
    ridge: np.ndarray,
    orientation: OrientationResult,
    pore_core: np.ndarray,
    config: FastDetectorConfig,
) -> np.ndarray:
    img = _normalize_image(image)
    ridge_u = _robust_unit(ridge)
    positive = ridge_u[ridge_u > 0]
    if not positive.size:
        return np.zeros_like(img, dtype=bool)
    low_thr = float(np.percentile(positive, config.ridge_percentile))
    high_thr = float(np.percentile(positive, config.high_ridge_percentile))
    local_mean = ndimage.gaussian_filter(img, 3.0, mode="reflect")
    locally_bright = img >= (local_mean - 0.03)
    coherent_or_strong = (orientation.coherency >= config.min_coherency) | (ridge_u >= high_thr)
    mask = (
        (ridge_u >= low_thr)
        & coherent_or_strong
        & locally_bright
        & ~pore_core
    )
    mask = closing(mask, disk(1))
    try:
        mask = remove_small_objects(mask, max_size=config.min_component_pixels - 1)
    except TypeError:
        mask = remove_small_objects(mask, min_size=config.min_component_pixels)
    return skeletonize(mask)


_NEIGHBORS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _order_component(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        return coords.astype(float)
    points = {tuple(map(int, p)): i for i, p in enumerate(coords)}
    adjacency: list[list[int]] = [[] for _ in range(len(coords))]
    for i, (r, c) in enumerate(coords):
        for dr, dc in _NEIGHBORS:
            j = points.get((int(r + dr), int(c + dc)))
            if j is not None:
                adjacency[i].append(j)
    endpoints = [i for i, n in enumerate(adjacency) if len(n) <= 1]
    start = endpoints[0] if endpoints else 0
    order = [start]
    visited = {start}
    previous = -1
    current = start
    while len(order) < len(coords):
        options = [j for j in adjacency[current] if j != previous and j not in visited]
        if not options:
            # Loops and tiny disconnected artifacts: continue from nearest unvisited pixel.
            remaining = np.array([i for i in range(len(coords)) if i not in visited], dtype=int)
            if not len(remaining):
                break
            here = coords[current]
            current = int(remaining[np.argmin(np.sum((coords[remaining] - here) ** 2, axis=1))])
            previous = -1
            visited.add(current)
            order.append(current)
            continue
        nxt = options[0]
        previous, current = current, nxt
        visited.add(current)
        order.append(current)
    return coords[np.asarray(order, dtype=int)].astype(float)


def _extract_paths(skeleton: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    degree = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skeleton
    junctions = skeleton & (degree > 2)
    segments = skeleton & ~junctions
    labels, count = ndimage.label(segments, structure=np.ones((3, 3), np.uint8))
    paths: list[np.ndarray] = []
    for label_id in range(1, count + 1):
        coords = np.argwhere(labels == label_id)
        if len(coords) < int(min_pixels):
            continue
        ordered = _order_component(coords)
        if len(ordered) >= int(min_pixels):
            paths.append(ordered)
    return paths


def _path_samples(path: np.ndarray, spacing: float) -> dict[str, np.ndarray] | None:
    if len(path) < 3:
        return None
    r = ndimage.gaussian_filter1d(path[:, 0], 1.2, mode="nearest")
    c = ndimage.gaussian_filter1d(path[:, 1], 1.2, mode="nearest")
    ds = np.hypot(np.diff(r), np.diff(c))
    arc = np.r_[0.0, np.cumsum(ds)]
    total = float(arc[-1])
    if total < max(2.0 * spacing, 6.0):
        return None
    targets = np.arange(0.5 * spacing, total, spacing, dtype=float)
    if not len(targets):
        targets = np.asarray([0.5 * total])
    rr = np.interp(targets, arc, r)
    cc = np.interp(targets, arc, c)
    dr = np.gradient(r, arc, edge_order=1)
    dc = np.gradient(c, arc, edge_order=1)
    drr = np.interp(targets, arc, dr)
    dcc = np.interp(targets, arc, dc)
    norm = np.hypot(drr, dcc) + 1e-9
    drr /= norm
    dcc /= norm
    path_angle = wrap90(np.rad2deg(np.arctan2(-drr, dcc)))
    weights = np.full(len(targets), spacing, dtype=float)
    if len(weights):
        weights[0] = min(spacing, targets[0] + 0.5 * spacing)
        weights[-1] = min(spacing, total - targets[-1] + 0.5 * spacing)
    return {
        "r": rr,
        "c": cc,
        "path_angle": np.asarray(path_angle, float),
        "arc": targets,
        "length_weight": weights,
        "total_length": np.asarray([total]),
    }


def _sample(field: np.ndarray, c: np.ndarray, r: np.ndarray, order: int = 1) -> np.ndarray:
    coords = np.vstack([np.asarray(r, float).ravel(), np.asarray(c, float).ravel()])
    values = ndimage.map_coordinates(field, coords, order=order, mode="nearest")
    return values.reshape(np.asarray(c).shape)


def _blend_directions(path_angle: np.ndarray, local_theta: np.ndarray, coherency: np.ndarray) -> np.ndarray:
    path_angle = np.asarray(path_angle, float)
    local_theta = np.asarray(local_theta, float)
    coherency = np.clip(np.asarray(coherency, float), 0.0, 1.0)
    error = axial_error_deg(path_angle, local_theta)
    orient_weight = np.where(error <= 42.0, 0.15 + 0.35 * coherency, 0.0)
    path_weight = 1.0 - orient_weight
    c = path_weight * np.cos(np.deg2rad(2.0 * path_angle)) + orient_weight * np.cos(np.deg2rad(2.0 * local_theta))
    s = path_weight * np.sin(np.deg2rad(2.0 * path_angle)) + orient_weight * np.sin(np.deg2rad(2.0 * local_theta))
    return wrap90(np.rad2deg(0.5 * np.arctan2(s, c)))


def _direction_segments(
    angles: np.ndarray,
    threshold_deg: float,
    persistence: int,
) -> np.ndarray:
    if len(angles) == 0:
        return np.empty(0, int)
    doubled = np.unwrap(np.deg2rad(2.0 * np.asarray(angles, float)))
    smooth = np.rad2deg(ndimage.gaussian_filter1d(doubled, 1.0, mode="nearest")) / 2.0
    ids = np.zeros(len(smooth), dtype=int)
    seg = 0
    reference = float(np.median(smooth[: min(3, len(smooth))]))
    i = 1
    while i < len(smooth):
        window = smooth[i : min(len(smooth), i + max(1, persistence))]
        if len(window) >= max(1, persistence) and abs(float(np.median(window)) - reference) >= threshold_deg:
            seg += 1
            reference = float(np.median(window))
        else:
            reference = 0.88 * reference + 0.12 * float(smooth[i])
        ids[i] = seg
        i += 1
    # Fill any gaps caused by split assignment and keep IDs contiguous.
    for i in range(1, len(ids)):
        if ids[i] < ids[i - 1]:
            ids[i] = ids[i - 1]
    return ids


def _first_edges_vectorized(
    gy: np.ndarray,
    gx: np.ndarray,
    gm: np.ndarray,
    centers_r: np.ndarray,
    centers_c: np.ndarray,
    angles: np.ndarray,
    max_half_width: float,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = np.arange(1.0, float(max_half_width) + 0.01, 0.5, dtype=np.float32)
    radians = np.deg2rad(np.asarray(angles, float))
    nr = np.cos(radians)
    nc = np.sin(radians)

    outputs = []
    for sign in (-1.0, 1.0):
        rr = centers_r[:, None] + sign * nr[:, None] * distances[None, :]
        cc = centers_c[:, None] + sign * nc[:, None] * distances[None, :]
        mag = _sample(gm, cc, rr)
        gyv = _sample(gy, cc, rr)
        gxv = _sample(gx, cc, rr)
        outward = (gyv * nr[:, None] + gxv * nc[:, None]) * sign
        local_peak = mag >= (ndimage.maximum_filter1d(mag, size=5, axis=1, mode="nearest") - 1e-8)
        row_threshold = np.maximum(edge_threshold, 0.28 * np.nanmax(mag, axis=1))
        valid = (mag >= row_threshold[:, None]) & (outward < 0.0) & local_peak
        fallback = (mag >= row_threshold[:, None]) & local_peak
        missing = ~valid.any(axis=1)
        valid[missing] = fallback[missing]
        has = valid.any(axis=1)
        first = np.argmax(valid, axis=1)
        selected_distance = distances[first].astype(float)
        selected_strength = mag[np.arange(len(mag)), first].astype(float)
        selected_distance[~has] = np.nan
        selected_strength[~has] = np.nan
        outputs.append((selected_distance, selected_strength, rr, cc))
    left_d, left_s, left_rr, left_cc = outputs[0]
    right_d, right_s, right_rr, right_cc = outputs[1]
    return left_d, right_d, left_s, right_s, nr, nc


def _summaries(local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if local.empty:
        return pd.DataFrame(), pd.DataFrame()
    region_rows = []
    rep_rows = []
    for region_id, group in local.groupby("fiber_region_id", sort=False):
        widths = pd.to_numeric(group["width_px"], errors="coerce").dropna().to_numpy(float)
        if not len(widths):
            continue
        median = float(np.median(widths))
        region_rows.append({
            "fiber_region_id": region_id,
            "fiber_path_id": group["fiber_path_id"].iloc[0],
            "sample_count": int(len(widths)),
            "median_width_px": median,
            "min_width_px": float(np.min(widths)),
            "max_width_px": float(np.max(widths)),
            "p10_width_px": float(np.percentile(widths, 10)),
            "p90_width_px": float(np.percentile(widths, 90)),
            "path_length_px": float(pd.to_numeric(group["sample_length_px"], errors="coerce").fillna(0).sum()),
            "direction_segment_count": int(group["direction_segment_id"].nunique()),
        })
        rep_rows.append({
            "fiber_region_id": region_id,
            "subregion_id": 1,
            "representative_width_px": median,
            "fiber_count_weight": 1.0,
            "length_weight": float(pd.to_numeric(group["sample_length_px"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(region_rows), pd.DataFrame(rep_rows)


def detect_fibers_fast(
    image: np.ndarray,
    *,
    config: FastDetectorConfig | None = None,
    prefer_gpu: bool = True,
    progress_callback: Callable[[float, str], None] | None = None,
) -> FastDetectionResult:
    config = config or FastDetectorConfig()
    img = _normalize_image(image)
    _emit(progress_callback, 0.02, "공통 방향 지도 계산")
    orientation = analyze_orientation(
        img,
        sigma_px=config.orientation_sigma_px,
        min_coherency=0.06,
        min_energy_frac=0.008,
        prefer_gpu=prefer_gpu,
    )

    _emit(progress_callback, 0.22, "다중 크기 ridge 계산")
    ridge, ridge_scale, gy, gx, ridge_backend = compute_multiscale_bright_ridge(
        img,
        sigmas=config.ridge_sigmas,
        gradient_sigma=1.0,
        prefer_gpu=prefer_gpu,
    )
    ridge_u = _robust_unit(ridge)
    gm = np.hypot(gy, gx).astype(np.float32)

    _emit(progress_callback, 0.43, "Pore core와 centerline 후보 생성")
    pore_core = build_pore_core(img, orientation.energy, ridge_u)
    skeleton = _candidate_skeleton(img, ridge_u, orientation, pore_core, config)
    paths = _extract_paths(skeleton, config.min_component_pixels)

    _emit(progress_callback, 0.57, "방향 graph 경로 정리")
    sample_blocks: list[dict] = []
    for path_id, path in enumerate(paths, start=1):
        sample = _path_samples(path, config.sample_spacing_px)
        if sample is None or float(sample["total_length"][0]) < config.min_path_length_px:
            continue
        rr = sample["r"]
        cc = sample["c"]
        local_theta = _sample(orientation.theta, cc, rr)
        local_coh = np.clip(_sample(orientation.coherency, cc, rr), 0.0, 1.0)
        direction = _blend_directions(sample["path_angle"], local_theta, local_coh)
        segments = _direction_segments(
            direction,
            threshold_deg=config.direction_split_deg,
            persistence=config.direction_split_persistence,
        )
        for i in range(len(rr)):
            sample_blocks.append({
                "fiber_path_id": path_id,
                "fiber_region_id": path_id,
                "region_sample_index": i,
                "direction_segment_id": int(segments[i]),
                "center_y": float(rr[i]),
                "center_x": float(cc[i]),
                "path_direction_deg": float(sample["path_angle"][i]),
                "direction_deg": float(direction[i]),
                "local_orientation_deg": float(local_theta[i]),
                "local_coherency": float(local_coh[i]),
                "sample_length_px": float(sample["length_weight"][i]),
                "path_arc_px": float(sample["arc"][i]),
            })
            if len(sample_blocks) >= config.max_measurements * 2:
                break
        if len(sample_blocks) >= config.max_measurements * 2:
            break

    if not sample_blocks:
        empty = pd.DataFrame()
        summary = {
            "algorithm": "fast_direction_graph_v1",
            "accepted_measurements": 0,
            "fiber_paths": 0,
            "direction_segments": 0,
            "pore_fraction": float(pore_core.mean()),
            "compute_backend": orientation.compute_backend,
            "ridge_compute_backend": ridge_backend.name,
        }
        return FastDetectionResult(empty, empty, empty, empty, orientation, pore_core, ridge_u, ridge_scale, summary)

    samples = pd.DataFrame(sample_blocks)
    r = samples["center_y"].to_numpy(float)
    c = samples["center_x"].to_numpy(float)
    angles = samples["direction_deg"].to_numpy(float)
    finite_gm = gm[np.isfinite(gm)]
    edge_threshold = float(np.percentile(finite_gm, 66.0)) if finite_gm.size else 0.0

    _emit(progress_callback, 0.69, "법선 방향 edge를 일괄 측정")
    left, right, left_strength, right_strength, nr, nc = _first_edges_vectorized(
        gy, gx, gm, r, c, angles, config.max_half_width_px, edge_threshold,
    )
    width = left + right
    scale_at_center = _sample(ridge_scale, c, r)
    ridge_at_center = np.clip(_sample(ridge_u, c, r), 0.0, 1.0)
    energy_at_center = np.clip(_sample(_robust_unit(orientation.energy, 99.0), c, r), 0.0, 1.0)
    pore_at_center = _sample(pore_core.astype(np.float32), c, r, order=0) > 0.5

    # Sample pore evidence along the candidate thickness chord in one vectorized pass.
    t = np.linspace(-1.0, 1.0, 25, dtype=float)
    signed_distance = np.where(t[None, :] < 0, (-t[None, :]) * left[:, None], t[None, :] * right[:, None])
    sign = np.where(t[None, :] < 0, -1.0, 1.0)
    chord_r = r[:, None] + sign * nr[:, None] * np.abs(signed_distance)
    chord_c = c[:, None] + sign * nc[:, None] * np.abs(signed_distance)
    pore_fraction = _sample(pore_core.astype(np.float32), chord_c, chord_r, order=0).mean(axis=1)

    expected_max = np.maximum(8.0, 7.0 * scale_at_center)
    symmetry = np.maximum(left, right) / np.maximum(np.minimum(left, right), 1e-6)
    edge_norm = np.nanpercentile(finite_gm, 98.0) if finite_gm.size else 1.0
    edge_score = np.clip(0.5 * (left_strength + right_strength) / max(float(edge_norm), 1e-9), 0.0, 1.0)
    symmetry_score = np.exp(-((np.log(np.maximum(symmetry, 1.0)) / 0.75) ** 2))
    orient_error = axial_error_deg(samples["path_direction_deg"].to_numpy(float), samples["local_orientation_deg"].to_numpy(float))
    orientation_score = np.exp(-((orient_error / 32.0) ** 2)) * (0.35 + 0.65 * samples["local_coherency"].to_numpy(float))
    confidence = np.clip(
        0.31 * ridge_at_center
        + 0.24 * edge_score
        + 0.15 * symmetry_score
        + 0.16 * orientation_score
        + 0.08 * energy_at_center
        + 0.06 * (1.0 - np.clip(pore_fraction / max(config.pore_reject_fraction, 1e-6), 0.0, 1.0)),
        0.0,
        1.0,
    )

    valid = (
        np.isfinite(width)
        & (width >= config.min_width_px)
        & (width <= config.max_width_px)
        & (width <= expected_max)
        & (symmetry <= 3.5)
        & ~pore_at_center
        & (pore_fraction <= config.pore_reject_fraction)
        & (confidence >= config.min_confidence)
    )

    accepted = samples.loc[valid].copy().reset_index(drop=True)
    rejected = samples.loc[~valid].copy().reset_index(drop=True)
    accepted["width_px"] = width[valid]
    accepted["left_width_px"] = left[valid]
    accepted["right_width_px"] = right[valid]
    accepted["x1"] = accepted["center_x"].to_numpy(float) - nc[valid] * left[valid]
    accepted["y1"] = accepted["center_y"].to_numpy(float) - nr[valid] * left[valid]
    accepted["x2"] = accepted["center_x"].to_numpy(float) + nc[valid] * right[valid]
    accepted["y2"] = accepted["center_y"].to_numpy(float) + nr[valid] * right[valid]
    accepted["xm"] = accepted["center_x"]
    accepted["ym"] = accepted["center_y"]
    accepted["orientation_error_deg"] = orient_error[valid]
    accepted["orientation_score"] = np.clip(orientation_score[valid], 0.0, 1.0)
    accepted["ridge_score"] = ridge_at_center[valid]
    accepted["edge_score"] = edge_score[valid]
    accepted["pore_fraction"] = pore_fraction[valid]
    accepted["confidence"] = confidence[valid]
    accepted["sem_agreement"] = confidence[valid]
    accepted["bundle_score"] = np.clip(pore_fraction[valid] / max(config.pore_reject_fraction, 1e-6), 0.0, 1.0)
    accepted["grade"] = np.where(confidence[valid] >= 0.68, "A", np.where(confidence[valid] >= 0.46, "B", "C"))
    accepted["detector"] = "fast_direction_graph"

    # Re-index samples inside each path after rejected candidates are removed.
    if not accepted.empty:
        accepted["region_sample_index"] = accepted.groupby("fiber_region_id").cumcount()
        accepted = accepted.sort_values(["fiber_region_id", "path_arc_px"]).reset_index(drop=True)
        if len(accepted) > config.max_measurements:
            accepted = accepted.nlargest(config.max_measurements, "confidence").sort_values(
                ["fiber_region_id", "path_arc_px"]
            ).reset_index(drop=True)

    if not rejected.empty:
        rejected["reason"] = "edge_or_pore_rejected"
        rejected["width_px"] = width[~valid]
        rejected["confidence"] = confidence[~valid]
        rejected["pore_fraction"] = pore_fraction[~valid]

    regions, representatives = _summaries(accepted)
    summary = {
        "algorithm": "fast_direction_graph_v1",
        "accepted_measurements": int(len(accepted)),
        "candidate_measurements": int(len(samples)),
        "rejected_measurements": int(len(rejected)),
        "fiber_paths": int(accepted["fiber_path_id"].nunique()) if not accepted.empty else 0,
        "direction_segments": int(accepted[["fiber_path_id", "direction_segment_id"]].drop_duplicates().shape[0]) if not accepted.empty else 0,
        "median_width_px": float(accepted["width_px"].median()) if not accepted.empty else float("nan"),
        "pore_fraction": float(pore_core.mean()),
        "skeleton_pixels": int(skeleton.sum()),
        "compute_backend": orientation.compute_backend,
        "compute_backend_detail": orientation.compute_backend_detail,
        "ridge_compute_backend": ridge_backend.name,
        "ridge_compute_backend_detail": ridge_backend.detail,
    }
    _emit(progress_callback, 0.96, "Fiber 경로와 방향 구간 요약")
    return FastDetectionResult(
        measurements=accepted,
        regions=regions,
        representatives=representatives,
        candidates=rejected,
        orientation=orientation,
        pore_core=pore_core,
        ridge_response=ridge_u,
        ridge_scale=ridge_scale,
        summary=summary,
    )
