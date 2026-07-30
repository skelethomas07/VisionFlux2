from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.color import hsv2rgb


@dataclass(frozen=True)
class OrientationResult:
    theta: np.ndarray
    coherency: np.ndarray
    energy: np.ndarray
    gate: np.ndarray
    color_map: np.ndarray
    histogram_centers_deg: np.ndarray
    histogram_fraction: np.ndarray
    dominant_direction_deg: float
    dominant_coherency: float
    order_parameter: float
    gated_fraction: float
    sigma_px: float


def wrap90(angle):
    """Wrap axial directions to [-90, 90)."""
    return (np.asarray(angle, dtype=float) + 90.0) % 180.0 - 90.0


def axial_error_deg(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0).astype(np.float32)


def _structure_tensor(image: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = _normalize_image(image)
    derivative_sigma = max(0.8, 0.45 * float(sigma_px))
    gy = ndimage.gaussian_filter(img, derivative_sigma, order=(1, 0), mode="reflect")
    gx = ndimage.gaussian_filter(img, derivative_sigma, order=(0, 1), mode="reflect")
    tensor_sigma = max(1.0, float(sigma_px))
    jyy = ndimage.gaussian_filter(gy * gy, tensor_sigma, mode="reflect")
    jxy = ndimage.gaussian_filter(gy * gx, tensor_sigma, mode="reflect")
    jxx = ndimage.gaussian_filter(gx * gx, tensor_sigma, mode="reflect")
    return jyy.astype(np.float32), jxy.astype(np.float32), jxx.astype(np.float32)


def _axial_stats(theta: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    t = np.deg2rad(np.asarray(theta, dtype=float).ravel()) * 2.0
    w = np.asarray(weights, dtype=float).ravel()
    good = np.isfinite(t) & np.isfinite(w) & (w > 0)
    if not good.any():
        return float("nan"), float("nan")
    t = t[good]
    w = w[good]
    w = w / w.sum()
    c = float(np.sum(w * np.cos(t)))
    s = float(np.sum(w * np.sin(t)))
    return float(wrap90(np.rad2deg(0.5 * np.arctan2(s, c)))), float(np.hypot(c, s))


def analyze_orientation(
    image: np.ndarray,
    sigma_px: float = 4.0,
    min_coherency: float = 0.20,
    min_energy_frac: float = 0.02,
    nbins: int = 180,
) -> OrientationResult:
    """Compute OrientationJ-compatible direction, coherency and weighted statistics."""
    jyy, jxy, jxx = _structure_tensor(image, sigma_px)
    energy = (jxx + jyy).astype(np.float32)
    theta = np.rad2deg(0.5 * np.arctan2(2.0 * jxy, jyy - jxx)).astype(np.float32)
    coherency = np.clip(
        np.sqrt((jyy - jxx) ** 2 + 4.0 * jxy ** 2) / (energy + 1e-20),
        0.0,
        1.0,
    ).astype(np.float32)

    e_threshold = float(min_energy_frac) * float(np.nanmax(energy) if energy.size else 0.0)
    gate = (
        np.isfinite(theta)
        & np.isfinite(coherency)
        & (coherency >= float(min_coherency))
        & (energy >= e_threshold)
    )

    hist, edges = np.histogram(
        theta[gate], bins=int(nbins), range=(-90.0, 90.0), weights=coherency[gate]
    )
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])

    if gate.any():
        syy = float(jyy[gate].sum())
        sxy = float(jxy[gate].sum())
        sxx = float(jxx[gate].sum())
        dominant = float(wrap90(np.rad2deg(0.5 * np.arctan2(2.0 * sxy, syy - sxx))))
        dominant_coh = float(
            np.sqrt((syy - sxx) ** 2 + 4.0 * sxy ** 2) / (sxx + syy + 1e-20)
        )
        _, order_parameter = _axial_stats(theta[gate], coherency[gate])
    else:
        dominant = float("nan")
        dominant_coh = float("nan")
        order_parameter = float("nan")

    img = _normalize_image(image)
    hsv = np.empty((*img.shape, 3), dtype=np.float32)
    hsv[..., 0] = (theta % 180.0) / 180.0
    hsv[..., 1] = coherency
    hsv[..., 2] = img
    color_map = np.clip(hsv2rgb(hsv), 0.0, 1.0).astype(np.float32)

    return OrientationResult(
        theta=theta,
        coherency=coherency,
        energy=energy,
        gate=gate,
        color_map=color_map,
        histogram_centers_deg=centers.astype(np.float32),
        histogram_fraction=hist.astype(np.float32),
        dominant_direction_deg=dominant,
        dominant_coherency=dominant_coh,
        order_parameter=order_parameter,
        gated_fraction=float(gate.mean()),
        sigma_px=float(sigma_px),
    )


def _sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coords = np.vstack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest")


def chord_tangent_direction_deg(x1, y1, x2, y2):
    """Return the fibre tangent; the supplied chord is the thickness normal."""
    chord_direction = np.rad2deg(np.arctan2(-(np.asarray(y2) - np.asarray(y1)), np.asarray(x2) - np.asarray(x1)))
    return wrap90(chord_direction + 90.0)


def annotate_measurement_directions(
    measurements: pd.DataFrame,
    orientation: OrientationResult,
) -> pd.DataFrame:
    if measurements is None or measurements.empty:
        return measurements.copy() if measurements is not None else pd.DataFrame()
    out = measurements.copy(deep=True)
    x1 = pd.to_numeric(out["x1"], errors="coerce").to_numpy(float)
    y1 = pd.to_numeric(out["y1"], errors="coerce").to_numpy(float)
    x2 = pd.to_numeric(out["x2"], errors="coerce").to_numpy(float)
    y2 = pd.to_numeric(out["y2"], errors="coerce").to_numpy(float)
    cx = pd.to_numeric(out.get("center_x", 0.5 * (x1 + x2)), errors="coerce").to_numpy(float)
    cy = pd.to_numeric(out.get("center_y", 0.5 * (y1 + y2)), errors="coerce").to_numpy(float)
    tangent = chord_tangent_direction_deg(x1, y1, x2, y2)
    local_theta = _sample(orientation.theta, cx, cy)
    local_coh = np.clip(_sample(orientation.coherency, cx, cy), 0.0, 1.0)
    error = axial_error_deg(tangent, local_theta)
    score = np.exp(-((error / 28.0) ** 2)) * (0.25 + 0.75 * local_coh)

    out["direction_deg"] = tangent
    out["local_orientation_deg"] = local_theta
    out["local_coherency"] = local_coh
    out["orientation_error_deg"] = error
    out["orientation_score"] = np.clip(score, 0.0, 1.0)
    base = pd.to_numeric(out.get("confidence", 0.5), errors="coerce").fillna(0.5).to_numpy(float)
    out["confidence"] = np.clip(0.72 * base + 0.28 * score, 0.0, 1.0)
    return out


def orientation_summary_dict(result: OrientationResult) -> dict:
    return {
        "dominant_direction_deg": result.dominant_direction_deg,
        "dominant_coherency": result.dominant_coherency,
        "orientation_order_parameter": result.order_parameter,
        "orientation_gated_fraction": result.gated_fraction,
        "orientation_sigma_px": result.sigma_px,
    }


def _component_order(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        return np.arange(len(coords))
    centered = coords.astype(float) - coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    return np.argsort(centered @ axis)


def _sample_component(coords: np.ndarray, spacing_px: float) -> np.ndarray:
    order = _component_order(coords)
    ordered = coords[order]
    if len(ordered) <= 1:
        return ordered
    keep = [0]
    accumulated = 0.0
    for i in range(1, len(ordered)):
        accumulated += float(np.hypot(*(ordered[i] - ordered[i - 1])))
        if accumulated >= spacing_px:
            keep.append(i)
            accumulated = 0.0
    if keep[-1] != len(ordered) - 1:
        keep.append(len(ordered) - 1)
    return ordered[np.asarray(keep, dtype=int)]


def _first_facing_edge(
    gy: np.ndarray,
    gx: np.ndarray,
    gm: np.ndarray,
    r: float,
    c: float,
    nr: float,
    nc: float,
    sign: float,
    max_half_width: float,
    edge_threshold: float,
) -> tuple[float, float] | None:
    distances = np.arange(1.0, float(max_half_width) + 0.01, 0.5)
    rr = r + sign * nr * distances
    cc = c + sign * nc * distances
    magnitudes = _sample(gm, cc, rr)
    gyv = _sample(gy, cc, rr)
    gxv = _sample(gx, cc, rr)
    facing = (gyv * nr + gxv * nc) * sign < 0.0
    valid = np.flatnonzero((magnitudes >= edge_threshold) & facing)
    if not len(valid):
        valid = np.flatnonzero(magnitudes >= edge_threshold)
    if not len(valid):
        return None
    # Prefer the first strong boundary so a neighboring parallel fibre is not absorbed.
    first_band = valid[distances[valid] <= distances[valid[0]] + 2.0]
    j = int(first_band[np.argmax(magnitudes[first_band])])
    return float(distances[j]), float(magnitudes[j])


def _profile_has_bundle(image: np.ndarray, r: float, c: float, nr: float, nc: float, left: float, right: float) -> bool:
    from scipy.signal import find_peaks

    ts = np.linspace(-left, right, max(12, int(round((left + right) * 2)) + 1))
    profile = _sample(image, c + nc * ts, r + nr * ts)
    if len(profile) < 8:
        return False
    smooth = ndimage.gaussian_filter1d(profile.astype(float), 1.0)
    span = float(np.ptp(smooth))
    if span < 0.05:
        return False
    peaks, props = find_peaks(smooth, prominence=max(0.05, 0.18 * span), distance=3)
    if len(peaks) < 2:
        return False
    strongest = peaks[np.argsort(props["prominences"])[-2:]]
    a, b = sorted(map(int, strongest))
    valley = float(np.min(smooth[a:b + 1]))
    shoulder = float(min(smooth[a], smooth[b]))
    return shoulder - valley >= max(0.06, 0.22 * span)


def orientation_guided_rescue(
    image: np.ndarray,
    orientation: OrientationResult,
    existing_measurements: pd.DataFrame | None = None,
    sample_spacing_px: float = 7.0,
    max_candidates: int = 900,
    max_half_width_px: float = 30.0,
) -> pd.DataFrame:
    """Add missed measurements from coherent ridge centre-lines and opposing edges.

    This branch is intentionally conservative: it uses the OrientationJ tangent at each
    centre, finds the first facing gradient on both normal sides, and rejects profiles
    that contain a persistent peak-valley-peak bundle signature.
    """
    from skimage.filters import sato
    from skimage.morphology import closing, disk, remove_small_objects, skeletonize

    img = _normalize_image(image)
    response = np.nan_to_num(
        sato(img, sigmas=(1.0, 1.6, 2.4, 3.6, 5.2, 7.0), black_ridges=False, mode="reflect")
    )
    response = (response - response.min()) / (float(np.ptp(response)) + 1e-9)
    finite = np.isfinite(response)
    if not finite.any() or float(response.max()) <= 0:
        return pd.DataFrame()
    ridge_threshold = float(np.percentile(response[finite], 84.0))
    intensity_threshold = float(np.percentile(img, 32.0))
    mask = (
        (response >= ridge_threshold)
        & (img >= intensity_threshold)
        & (orientation.coherency >= 0.16)
        & orientation.gate
    )
    mask = closing(mask, disk(1))
    try:
        mask = remove_small_objects(mask, max_size=17)
    except TypeError:
        mask = remove_small_objects(mask, min_size=18)
    skeleton = skeletonize(mask)
    labels, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))

    gy = ndimage.gaussian_filter(img, 1.2, order=(1, 0), mode="reflect")
    gx = ndimage.gaussian_filter(img, 1.2, order=(0, 1), mode="reflect")
    gm = np.hypot(gy, gx)
    edge_threshold = float(np.percentile(gm[np.isfinite(gm)], 68.0))

    existing = existing_measurements if existing_measurements is not None else pd.DataFrame()
    if not existing.empty:
        ex_x = pd.to_numeric(existing.get("center_x"), errors="coerce").to_numpy(float)
        ex_y = pd.to_numeric(existing.get("center_y"), errors="coerce").to_numpy(float)
        ex_w = pd.to_numeric(existing.get("width_px"), errors="coerce").to_numpy(float)
        ex_dir = pd.to_numeric(existing.get("direction_deg", np.nan), errors="coerce").to_numpy(float)
    else:
        ex_x = ex_y = ex_w = ex_dir = np.empty(0, float)

    rows: list[dict] = []
    for component_id in range(1, count + 1):
        coords = np.argwhere(labels == component_id)
        if len(coords) < 10:
            continue
        sampled = _sample_component(coords, max(2.0, float(sample_spacing_px)))
        sample_index = 0
        for r_i, c_i in sampled:
            r = float(r_i)
            c = float(c_i)
            coh = float(orientation.coherency[r_i, c_i])
            if coh < 0.18:
                continue
            angle = float(orientation.theta[r_i, c_i])
            angle_rad = math.radians(angle)
            nr, nc = math.cos(angle_rad), math.sin(angle_rad)
            left = _first_facing_edge(gy, gx, gm, r, c, nr, nc, -1.0, max_half_width_px, edge_threshold)
            right = _first_facing_edge(gy, gx, gm, r, c, nr, nc, +1.0, max_half_width_px, edge_threshold)
            if left is None or right is None:
                continue
            wl, gl = left
            wr, gr = right
            width = wl + wr
            if not (2.5 <= width <= 2.0 * max_half_width_px):
                continue
            if max(wl, wr) / max(min(wl, wr), 1e-6) > 3.2:
                continue
            # Direction must remain stable around the candidate centre.
            tr, tc = -math.sin(angle_rad), math.cos(angle_rad)
            local_theta = _sample(
                orientation.theta,
                np.asarray([c - 3 * tc, c, c + 3 * tc]),
                np.asarray([r - 3 * tr, r, r + 3 * tr]),
            )
            if float(np.nanmax(axial_error_deg(local_theta, angle))) > 28.0:
                continue
            if _profile_has_bundle(img, r, c, nr, nc, wl, wr):
                continue
            if len(ex_x):
                d = np.hypot(ex_x - c, ex_y - r)
                near = np.flatnonzero(d <= 3.5)
                if len(near):
                    same = (
                        np.abs(ex_w[near] - width) / np.maximum(ex_w[near], width) <= 0.45
                    )
                    if np.isfinite(ex_dir[near]).any():
                        same &= axial_error_deg(ex_dir[near], angle) <= 30.0
                    if same.any():
                        continue
            x1 = c - nc * wl
            y1 = r - nr * wl
            x2 = c + nc * wr
            y2 = r + nr * wr
            edge_score = min(1.0, 0.5 * (gl + gr) / (float(np.percentile(gm, 98)) + 1e-9))
            confidence = float(np.clip(0.45 * response[r_i, c_i] + 0.35 * coh + 0.20 * edge_score, 0, 1))
            rows.append({
                "fiber_region_id": f"orientation-{component_id}",
                "region_sample_index": sample_index,
                "center_x": c,
                "center_y": r,
                "xm": c,
                "ym": r,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width_px": float(width),
                "direction_deg": angle,
                "local_orientation_deg": angle,
                "local_coherency": coh,
                "orientation_error_deg": 0.0,
                "orientation_score": float(np.clip(coh * edge_score, 0, 1)),
                "confidence": confidence,
                "sem_agreement": confidence,
                "bundle_score": 0.0,
                "grade": "O",
                "detector": "orientation_rescue",
            })
            sample_index += 1
            if len(rows) >= int(max_candidates):
                break
        if len(rows) >= int(max_candidates):
            break
    return pd.DataFrame(rows)
