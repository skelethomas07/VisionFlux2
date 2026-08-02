# ================================================================
# One-cell hybrid batch: thin + thick-rescue + curved-rescue fibre thickness
# Designed for Google Colab and dense SEM fibre images.
# Clear thin fibres are kept strict; thicker and curved fibres are rescued with
# OrientationJ direction fields and along-curve continuity. Ambiguous crossings are rejected.
# ================================================================

from pathlib import Path
import re, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage
from scipy.signal import find_peaks
from PIL import Image
import tifffile
from skimage.feature import canny
from skimage.morphology import closing, disk, thin, dilation

# ---------------------------------------------------------------- CONFIG
IMAGE_DIR = Path('/content/drive/MyDrive/CHEM FRONTIER/SEM 자료들')
OUTPUT_DIR = IMAGE_DIR.parent / f'{IMAGE_DIR.name}_edge_thickness_hybrid'

# Run exactly these five images, in this order.
TARGET_FILES = ['2-10.jpg', '2-11.jpg', '2-19.jpg', '2-20.jpg', '2-21.jpg']

# Optional physical calibration. Fill only images whose nm/px is known.
NM_PER_PX = {
    # '2-10.jpg': 5.0,
}

SHOW_INLINE = False
SAVE_DPI = 170
AUTO_CROP_DATABAR = True

# Common continuous edge construction. Canny includes gradient-direction NMS
# and hysteresis linking; these curves provide the source edge and local tangent.
EDGE_SIGMA_PX = 1.25
CANNY_LOW_QUANTILE = 0.55
CANNY_HIGH_QUANTILE = 0.85
EDGE_CLOSE_RADIUS_PX = 1
MIN_EDGE_BRANCH_PX = 18
MAX_JOIN_TURN_DEG = 28.0
JOIN_JUNCTION_RADIUS_PX = 2
BRIDGE_GAP_PX = 3.0
BRIDGE_TURN_DEG = 18.0

# A denser one-pixel candidate mask is used only when looking for the opposite
# side of thick/curved fibres. It is not independently traced into fibres.
DENSE_EDGE_GRAD_PERCENTILE = 82.0
DENSE_EDGE_MIN_COHERENCY = 0.12
DENSE_EDGE_MIN_ENERGY_FRAC = 0.01
DENSE_EDGE_HIT_TOLERANCE_PX = 1.55

# Shared profile sampling.
PROFILE_STEP_PX = 0.5
START_EDGE_REFINE_PX = 1.5
PROFILE_OUTSIDE_PX = 3.0
PROFILE_EDGE_MARGIN_PX = 1.25
BORDER_EXCLUSION_PX = 4
JUNCTION_EXCLUSION_PX = 4
SAMPLE_SPACING_PX = 3.0

# Branch 1 — keep the current strict mechanism for thin/clear fibres.
THIN_MIN_WIDTH_PX = 2.5
THIN_MAX_WIDTH_PX = 18.0
THIN_EDGE_HIT_TOLERANCE_PX = 1.35
THIN_MAX_TANGENT_MISMATCH_DEG = 25.0
THIN_MIN_PROFILE_CONTRAST = 0.045
THIN_MAX_BACKGROUND_DIFFERENCE = 0.18
THIN_INTERIOR_PEAK_RATIO_MAX = 0.82
THIN_MAX_CANDIDATES = 3
THIN_MIN_PAIR_SCORE = 0.48
THIN_MIN_RUN_SAMPLES = 8
THIN_MIN_RUN_LENGTH_PX = 24.0
THIN_MAX_RUN_ROBUST_CV = 0.22

# Branch 2 — thicker, mostly straight fibres. Multiple normal-profile peaks are
# retained; a stable outer boundary is selected by width/position continuity.
THICK_MIN_WIDTH_PX = 10.0              # overlaps thin branch; merge removes duplicates
THICK_MAX_WIDTH_PX = None              # None -> min(70 px, 14% of smaller dimension)
THICK_MAX_LOCAL_TURN_DEG_PER_PX = 0.55
THICK_MAX_TANGENT_MISMATCH_DEG = 32.0
THICK_MIN_PROFILE_CONTRAST = 0.030
THICK_MAX_BACKGROUND_DIFFERENCE = 0.26
THICK_MIN_ORIENTATION_PROFILE_SCORE = 0.54
THICK_MAX_CANDIDATES = 9
THICK_MIN_PAIR_SCORE = 0.43
THICK_MIN_RUN_SAMPLES = 6
THICK_MIN_RUN_LENGTH_PX = 24.0
THICK_MAX_RUN_ROBUST_CV = 0.20

# Branch 3 — curved fibres. Local tangent/normal are updated at every sample;
# orientation may rotate along the curve but must do so smoothly.
CURVED_MIN_WIDTH_PX = 2.5
CURVED_MAX_WIDTH_PX = None             # same automatic cap as thick branch
CURVED_MIN_LOCAL_TURN_DEG_PER_PX = 0.18
CURVED_MIN_RUN_TOTAL_TURN_DEG = 8.0
CURVED_MAX_TANGENT_MISMATCH_DEG = 40.0
CURVED_MIN_PROFILE_CONTRAST = 0.030
CURVED_MAX_BACKGROUND_DIFFERENCE = 0.27
CURVED_MAX_CANDIDATES = 8
CURVED_MIN_PAIR_SCORE = 0.42
CURVED_MIN_RUN_SAMPLES = 6
CURVED_MIN_RUN_LENGTH_PX = 18.0
CURVED_MAX_RUN_ROBUST_CV = 0.25

# Continuity and merge.
MAX_WIDTH_JUMP_FRAC = 0.34
CONTINUITY_WEIGHT = {'thin': 1.8, 'thick': 2.8, 'curved': 2.2}
MOTION_WEIGHT = {'thin': 0.35, 'thick': 0.65, 'curved': 0.55}
MAX_EMPTY_GAP_SAMPLES = {'thin': 0, 'thick': 1, 'curved': 1}
DEDUP_ENDPOINT_BIN_PX = 2.0
MERGE_MIDPOINT_RADIUS_PX = 2.8
MERGE_MAX_CHORD_ANGLE_DEG = 15.0
MERGE_MAX_WIDTH_REL_DIFF = 0.28
METHOD_PRIORITY = {'thin': 3, 'curved': 2, 'thick': 1}

SUPPORTED = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
N8 = np.ones((3, 3), np.uint8)
N8[1, 1] = 0

# --------------------------------------------------------------- I/O + fields
def natural_key(p):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r'(\d+)', Path(p).name)]


def load_gray(path):
    p = Path(path)
    a = tifffile.imread(p) if p.suffix.lower() in {'.tif', '.tiff'} else np.asarray(Image.open(p))
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3].astype(np.float32)
        a = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif a.ndim == 3:
        a = a[a.shape[0] // 2]
    a = np.squeeze(np.asarray(a, np.float32))
    if a.ndim != 2:
        raise ValueError(f'{p.name}: expected a 2-D image, got {a.shape}')
    good = np.isfinite(a)
    if not good.any():
        raise ValueError(f'{p.name}: no finite pixels')
    if not good.all():
        a[~good] = np.median(a[good])
    return a


def detect_databar(a, flat_frac=0.35, search_frac=0.30):
    h = a.shape[0]
    start = int(h * (1 - search_frac))
    r = a[start:]
    if r.size == 0:
        return h
    q = np.clip((r - r.min()) / max(float(np.ptp(r)), 1e-6) * 255, 0, 255).astype(np.uint8)
    flat = []
    for row in q:
        counts = np.bincount(row, minlength=256)
        flat.append(counts[np.argsort(counts)[-2:]].sum() / row.size)
    flat = np.asarray(flat)
    banner = flat > flat_frac
    if not banner.any():
        return h
    non_banner = np.flatnonzero(~banner)
    cut = start + (non_banner[-1] + 1 if non_banner.size else 0)
    return int(cut) if cut > h * 0.5 else h


def normalize_image(a):
    lo, hi = np.percentile(a[np.isfinite(a)], [0.5, 99.5])
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def prepare_image(path):
    raw = load_gray(path)
    cut = detect_databar(raw) if AUTO_CROP_DATABAR else raw.shape[0]
    return normalize_image(raw[:cut]), cut


def orientation_fields(img):
    # Gaussian derivatives + OrientationJ-compatible structure tensor convention.
    gy = ndimage.gaussian_filter(img, EDGE_SIGMA_PX, order=(1, 0), mode='reflect')
    gx = ndimage.gaussian_filter(img, EDGE_SIGMA_PX, order=(0, 1), mode='reflect')
    ts = max(2.0, 2.0 * EDGE_SIGMA_PX)
    Jyy = ndimage.gaussian_filter(gy * gy, ts, mode='reflect')
    Jxy = ndimage.gaussian_filter(gy * gx, ts, mode='reflect')
    Jxx = ndimage.gaussian_filter(gx * gx, ts, mode='reflect')
    energy = Jxx + Jyy
    theta = np.rad2deg(0.5 * np.arctan2(2.0 * Jxy, Jyy - Jxx)).astype(np.float32)
    coh = (np.sqrt((Jyy - Jxx) ** 2 + 4.0 * Jxy ** 2) /
           (energy + 1e-12)).astype(np.float32)
    grad = np.hypot(gy, gx).astype(np.float32)
    return gy.astype(np.float32), gx.astype(np.float32), grad, theta, coh, energy.astype(np.float32)


def build_continuous_edges(img, coh, energy):
    # Canny = Gaussian gradient + gradient-normal NMS + double threshold + hysteresis.
    edge = canny(img, sigma=EDGE_SIGMA_PX,
                 low_threshold=CANNY_LOW_QUANTILE,
                 high_threshold=CANNY_HIGH_QUANTILE,
                 use_quantiles=True)
    # Coherency is not a hard gate. Canny continuity + later pair/run tests decide.
    if EDGE_CLOSE_RADIUS_PX > 0:
        edge = closing(edge, disk(EDGE_CLOSE_RADIUS_PX))
    edge = thin(edge)
    edge = remove_small_components(edge, max(4, MIN_EDGE_BRANCH_PX // 2))
    return edge.astype(bool)


def nonmaximum_suppression(grad_mag, dy, dx, step=1.0):
    """Keep local gradient maxima along a supplied normal direction field."""
    H, W = grad_mag.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.asarray(grad_mag, np.float32)
    out = np.ones((H, W), bool)
    for sign in (+step, -step):
        coords = np.stack([yy + sign * dy, xx + sign * dx])
        neighbour = ndimage.map_coordinates(g, coords, order=1, mode='nearest').reshape(H, W)
        out &= g >= neighbour
    return out


def build_dense_edge_candidates(grad, gy, gx, coh, energy):
    """Dense one-pixel candidates used only as opposite-edge evidence."""
    finite = np.isfinite(grad) & np.isfinite(coh) & np.isfinite(energy)
    if not finite.any():
        return np.zeros_like(grad, bool), np.nan
    threshold = float(np.percentile(grad[finite], DENSE_EDGE_GRAD_PERCENTILE))
    gsafe = np.maximum(grad, 1e-12)
    ny, nx = gy / gsafe, gx / gsafe
    nms = nonmaximum_suppression(grad, ny, nx)
    emax = float(np.max(energy[finite]))
    mask = (finite & nms & (grad >= threshold) &
            (coh >= DENSE_EDGE_MIN_COHERENCY) &
            (energy >= DENSE_EDGE_MIN_ENERGY_FRAC * max(emax, 1e-12)))
    return mask.astype(bool), threshold


# ------------------------------------------------------------ edge graph + tracing
def remove_small_components(mask, min_pixels):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_pixels)
    keep[0] = False
    return keep[lab]


def neighbours(sk):
    return ndimage.convolve(sk.astype(np.uint8), N8, mode='constant') * sk


def order_path(pixel_set):
    adj = {p: [] for p in pixel_set}
    for y, x in pixel_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    q = (y + dy, x + dx)
                    if q in adj:
                        adj[(y, x)].append(q)
    ends = [p for p in pixel_set if len(adj[p]) <= 1]
    start = ends[0] if ends else next(iter(pixel_set))
    out, seen, cur = [start], {start}, start
    while True:
        nxt = [q for q in adj[cur] if q not in seen]
        if not nxt:
            break
        # Prefer the continuation that changes direction least when a pixel tie occurs.
        if len(out) >= 2:
            prev = np.asarray(out[-1], float) - np.asarray(out[-2], float)
            nxt.sort(key=lambda q: -float(np.dot(np.asarray(q, float) - np.asarray(cur, float), prev)))
        cur = nxt[0]
        seen.add(cur)
        out.append(cur)
    return np.asarray(out, dtype=np.float32)


def skeleton_branches(sk):
    nb = neighbours(sk)
    junction = sk & (nb >= 3)
    jlab, _ = ndimage.label(junction, structure=np.ones((3, 3), np.uint8))
    segments = sk & ~junction
    slab, n = ndimage.label(segments, structure=np.ones((3, 3), np.uint8))
    paths = []
    for lab in range(1, n + 1):
        ys, xs = np.where(slab == lab)
        if len(ys) >= 2:
            p = order_path(set(zip(ys.tolist(), xs.tolist())))
            if len(p) >= MIN_EDGE_BRANCH_PX:
                paths.append(p)
    return paths, jlab


def endpoint_tangent(path, end, k=8):
    p = np.asarray(path, float)
    if end == 0:
        a, b = p[min(k, len(p) - 1)], p[0]
    else:
        a, b = p[max(0, len(p) - 1 - k)], p[-1]
    v = b - a
    n = np.hypot(v[0], v[1])
    return v / n if n > 1e-9 else np.zeros(2)


def turn_between(path_a, end_a, path_b, end_b):
    ta = endpoint_tangent(path_a, end_a)
    tb = endpoint_tangent(path_b, end_b)
    return float(np.degrees(np.arccos(np.clip(-float(np.dot(ta, tb)), -1, 1))))


def pair_at_junctions(paths, jlab):
    dil = ndimage.grey_dilation(jlab, size=(2 * JOIN_JUNCTION_RADIUS_PX + 1,) * 2)
    by_junction = {}
    for i, p in enumerate(paths):
        for e, pt in ((0, tuple(np.round(p[0]).astype(int))),
                      (1, tuple(np.round(p[-1]).astype(int)))):
            lab = int(dil[pt])
            if lab:
                by_junction.setdefault(lab, []).append((i, e))
    partner = {}
    for ends in by_junction.values():
        candidates = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                A, B = ends[a], ends[b]
                if A[0] == B[0]:
                    continue
                turn = turn_between(paths[A[0]], A[1], paths[B[0]], B[1])
                if turn <= MAX_JOIN_TURN_DEG:
                    candidates.append((turn, A, B))
        candidates.sort(key=lambda z: z[0])
        used = set()
        for _, A, B in candidates:
            if A in used or B in used:
                continue
            partner[A], partner[B] = B, A
            used.update((A, B))
    return partner


def bridge_free_ends(paths, partner):
    free = [(i, e) for i in range(len(paths)) for e in (0, 1) if (i, e) not in partner]
    pos = {A: (paths[A[0]][0] if A[1] == 0 else paths[A[0]][-1]).astype(float) for A in free}
    candidates = []
    for ia in range(len(free)):
        for ib in range(ia + 1, len(free)):
            A, B = free[ia], free[ib]
            if A[0] == B[0]:
                continue
            delta = pos[B] - pos[A]
            gap = float(np.hypot(delta[0], delta[1]))
            if not (1e-6 < gap <= BRIDGE_GAP_PX):
                continue
            u = delta / gap
            ta = endpoint_tangent(paths[A[0]], A[1])
            tb = endpoint_tangent(paths[B[0]], B[1])
            aa = np.degrees(np.arccos(np.clip(float(np.dot(ta, u)), -1, 1)))
            ab = np.degrees(np.arccos(np.clip(float(np.dot(tb, -u)), -1, 1)))
            turn = max(aa, ab)
            if turn <= BRIDGE_TURN_DEG:
                candidates.append((turn + gap, A, B))
    candidates.sort(key=lambda z: z[0])
    used = set()
    for _, A, B in candidates:
        if A in used or B in used or A in partner or B in partner:
            continue
        partner[A], partner[B] = B, A
        used.update((A, B))
    return partner


def trace_edge_curves(edge):
    paths, jlab = skeleton_branches(edge)
    if not paths:
        return [], jlab
    partner = pair_at_junctions(paths, jlab)
    partner = bridge_free_ends(paths, partner)
    used = np.zeros(len(paths), bool)
    curves = []

    def chain_from(i, e):
        out = []
        while True:
            nxt = partner.get((i, e))
            if nxt is None or used[nxt[0]]:
                break
            j, je = nxt
            used[j] = True
            out.append((j, je))
            i, e = j, 1 - je
        return out

    order = sorted(range(len(paths)), key=lambda i: -len(paths[i]))
    for seed in order:
        if used[seed]:
            continue
        used[seed] = True
        fwd = chain_from(seed, 1)
        bwd = chain_from(seed, 0)
        pieces = []
        for j, je in reversed(bwd):
            q = paths[j]
            pieces.append(q if je == 1 else q[::-1])
        pieces.append(paths[seed])
        for j, je in fwd:
            q = paths[j]
            pieces.append(q if je == 0 else q[::-1])
        curve = np.vstack(pieces)
        if len(curve) >= MIN_EDGE_BRANCH_PX:
            curves.append(curve)
    return curves, jlab

# ----------------------------------------------------------- profiles + pair scoring
def resample_polyline(path, ds=1.0):
    p = np.asarray(path, float)
    if len(p) < 2:
        return p, np.zeros(len(p))
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    s = np.r_[0.0, np.cumsum(seg)]
    keep = np.r_[True, np.diff(s) > 1e-9]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < ds:
        return p, s
    sq = np.arange(0, s[-1] + 0.5 * ds, ds)
    q = np.c_[np.interp(sq, s, p[:, 0]), np.interp(sq, s, p[:, 1])]
    return q, sq


def sample_map(a, y, x):
    return float(ndimage.map_coordinates(a, [[y], [x]], order=1, mode='nearest')[0])


def axial_angle_diff_deg(a, b):
    """Smallest angle between unoriented axes, scalar or NumPy array."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180.0
    out = np.minimum(d, 180.0 - d)
    return float(out) if out.ndim == 0 else out


def axial_tangent_mismatch(theta_deg, ty, tx):
    t = np.deg2rad(theta_deg)
    oy, ox = -np.sin(t), np.cos(t)
    dot = np.clip(abs(oy * ty + ox * tx), 0, 1)
    return float(np.degrees(np.arccos(dot)))


def orientation_profile_score(theta_values, coherency_values, reference_theta_deg,
                              min_coherency=0.12, tolerance_deg=25.0):
    """Weighted agreement of OrientationJ directions with one axial reference."""
    theta_values = np.asarray(theta_values, float)
    coherency_values = np.asarray(coherency_values, float)
    valid = (np.isfinite(theta_values) & np.isfinite(coherency_values) &
             (coherency_values >= min_coherency))
    if valid.sum() < 2:
        return 0.0
    diff = axial_angle_diff_deg(theta_values[valid], reference_theta_deg)
    agreement = np.exp(-np.square(diff / max(float(tolerance_deg), 1e-6)))
    weights = np.clip(coherency_values[valid], 0.0, 1.0)
    return float(np.sum(weights * agreement) / max(np.sum(weights), 1e-12))


def curve_turn_rate_deg_per_px(path):
    """Robust median directed tangent rotation per pixel of arc length."""
    p, s = resample_polyline(path, 1.0)
    if len(p) < 5 or s[-1] <= 0:
        return 0.0
    sy = ndimage.gaussian_filter1d(p[:, 0], 2.0, mode='nearest')
    sx = ndimage.gaussian_filter1d(p[:, 1], 2.0, mode='nearest')
    dy, dx = np.gradient(sy), np.gradient(sx)
    angle = np.unwrap(np.arctan2(dy, dx))
    ds = np.maximum(np.gradient(s), 1e-6)
    rate = np.abs(np.gradient(angle)) / ds
    core = rate[2:-2] if len(rate) > 4 else rate
    return float(np.degrees(np.median(core))) if core.size else 0.0


def local_turn_rate_and_angle(sy, sx, arc):
    dy, dx = np.gradient(sy), np.gradient(sx)
    norm = np.hypot(dy, dx)
    ty = np.divide(dy, norm, out=np.zeros_like(dy), where=norm > 1e-6)
    tx = np.divide(dx, norm, out=np.zeros_like(dx), where=norm > 1e-6)
    directed = np.unwrap(np.arctan2(ty, tx))
    ds = np.maximum(np.gradient(arc), 1e-6)
    raw_rate = np.abs(np.gradient(directed)) / ds
    rate = ndimage.gaussian_filter1d(np.degrees(raw_rate), 3.0, mode='nearest')
    return ty, tx, directed, rate, norm > 1e-6


def branch_limits(mode, image_shape):
    automatic_max = min(70.0, 0.14 * min(image_shape))
    if mode == 'thin':
        return THIN_MIN_WIDTH_PX, min(THIN_MAX_WIDTH_PX, automatic_max)
    if mode == 'thick':
        return THICK_MIN_WIDTH_PX, automatic_max if THICK_MAX_WIDTH_PX is None else float(THICK_MAX_WIDTH_PX)
    if mode == 'curved':
        return CURVED_MIN_WIDTH_PX, automatic_max if CURVED_MAX_WIDTH_PX is None else float(CURVED_MAX_WIDTH_PX)
    raise ValueError(f'Unknown mode: {mode}')


def branch_parameters(mode):
    if mode == 'thin':
        return dict(max_mismatch=THIN_MAX_TANGENT_MISMATCH_DEG,
                    min_contrast=THIN_MIN_PROFILE_CONTRAST,
                    max_bg=THIN_MAX_BACKGROUND_DIFFERENCE,
                    max_candidates=THIN_MAX_CANDIDATES,
                    min_score=THIN_MIN_PAIR_SCORE,
                    hit_tolerance=THIN_EDGE_HIT_TOLERANCE_PX,
                    gradient_frac=0.30)
    if mode == 'thick':
        return dict(max_mismatch=THICK_MAX_TANGENT_MISMATCH_DEG,
                    min_contrast=THICK_MIN_PROFILE_CONTRAST,
                    max_bg=THICK_MAX_BACKGROUND_DIFFERENCE,
                    max_candidates=THICK_MAX_CANDIDATES,
                    min_score=THICK_MIN_PAIR_SCORE,
                    hit_tolerance=DENSE_EDGE_HIT_TOLERANCE_PX,
                    gradient_frac=0.22)
    if mode == 'curved':
        return dict(max_mismatch=CURVED_MAX_TANGENT_MISMATCH_DEG,
                    min_contrast=CURVED_MIN_PROFILE_CONTRAST,
                    max_bg=CURVED_MAX_BACKGROUND_DIFFERENCE,
                    max_candidates=CURVED_MAX_CANDIDATES,
                    min_score=CURVED_MIN_PAIR_SCORE,
                    hit_tolerance=DENSE_EDGE_HIT_TOLERANCE_PX,
                    gradient_frac=0.22)
    raise ValueError(f'Unknown mode: {mode}')


def candidate_pairs_at_sample(img, gy, gx, grad, theta, coh, search_edge_dist,
                              y, x, ty, tx, normal_y, normal_x,
                              min_width, max_width, grad_ref, mode):
    """Return possible opposite edges for one source-edge sample."""
    p = branch_parameters(mode)
    source_theta = sample_map(theta, y, x)
    source_coh = sample_map(coh, y, x)
    candidates = []

    for sign in (-1.0, 1.0):
        ny, nx = sign * normal_y, sign * normal_x
        tmin = -START_EDGE_REFINE_PX
        tmax = max_width + PROFILE_OUTSIDE_PX + 2.0
        ts = np.arange(tmin, tmax + 0.5 * PROFILE_STEP_PX,
                       PROFILE_STEP_PX, dtype=np.float32)
        yy, xx = y + ny * ts, x + nx * ts
        coords = np.vstack([yy, xx])
        valid = ((yy >= 1) & (yy < img.shape[0] - 1) &
                 (xx >= 1) & (xx < img.shape[1] - 1))
        if valid.sum() < 10:
            continue

        prof = ndimage.map_coordinates(img, coords, order=1, mode='nearest')
        py = ndimage.map_coordinates(gy, coords, order=1, mode='nearest')
        px = ndimage.map_coordinates(gx, coords, order=1, mode='nearest')
        theta_prof = ndimage.map_coordinates(theta, coords, order=1, mode='nearest')
        coh_prof = ndimage.map_coordinates(coh, coords, order=1, mode='nearest')
        dn = py * ny + px * nx
        amag = np.abs(dn)

        start_zone = np.where(np.abs(ts) <= START_EDGE_REFINE_PX)[0]
        if start_zone.size == 0:
            continue
        i0 = int(start_zone[np.argmax(amag[start_zone])])
        t0, d0 = float(ts[i0]), float(dn[i0])
        if abs(d0) < p['gradient_frac'] * grad_ref:
            continue

        peak_idx, _ = find_peaks(
            amag,
            distance=max(1, int(round(1.25 / PROFILE_STEP_PX))),
            prominence=max(1e-8, 0.08 * grad_ref if mode != 'thin' else 0.10 * grad_ref),
        )
        for ip in peak_idx:
            width = float(ts[ip] - t0)
            if not (min_width <= width <= max_width):
                continue
            y2, x2 = float(yy[ip]), float(xx[ip])
            if sample_map(search_edge_dist, y2, x2) > p['hit_tolerance']:
                continue

            d2 = float(dn[ip])
            if d0 * d2 >= 0 or min(abs(d0), abs(d2)) < p['gradient_frac'] * grad_ref:
                continue

            opposite_theta = sample_map(theta, y2, x2)
            opposite_coh = sample_map(coh, y2, x2)
            tangent_mismatch = axial_tangent_mismatch(opposite_theta, ty, tx)
            if tangent_mismatch > p['max_mismatch']:
                continue
            endpoint_theta_diff = axial_angle_diff_deg(source_theta, opposite_theta)
            endpoint_theta_score = float(np.exp(-((endpoint_theta_diff / 25.0) ** 2)))
            source_tangent_score = float(np.clip(
                1.0 - axial_tangent_mismatch(source_theta, ty, tx) / p['max_mismatch'], 0, 1))
            opposite_tangent_score = float(np.clip(
                1.0 - tangent_mismatch / p['max_mismatch'], 0, 1))
            tensor_tangent_score = 0.5 * (source_tangent_score + opposite_tangent_score)

            inside = (ts >= t0 + PROFILE_EDGE_MARGIN_PX) & (ts <= ts[ip] - PROFILE_EDGE_MARGIN_PX)
            bg1 = (ts >= t0 - PROFILE_OUTSIDE_PX) & (ts <= t0 - 0.75)
            bg2 = (ts >= ts[ip] + 0.75) & (ts <= ts[ip] + PROFILE_OUTSIDE_PX)
            if inside.sum() < 2 or bg1.sum() < 2 or bg2.sum() < 2:
                continue
            inside_level = float(np.median(prof[inside]))
            outside1 = float(np.median(prof[bg1]))
            outside2 = float(np.median(prof[bg2]))
            outside_level = 0.5 * (outside1 + outside2)
            contrast = abs(inside_level - outside_level)
            bg_difference = abs(outside1 - outside2)
            if contrast < p['min_contrast'] or bg_difference > p['max_bg']:
                continue

            interior_idx = np.where((ts >= t0 + 2.0 * PROFILE_EDGE_MARGIN_PX) &
                                    (ts <= ts[ip] - 2.0 * PROFILE_EDGE_MARGIN_PX))[0]
            interior_ratio = 0.0
            orientation_score = 0.0
            if interior_idx.size:
                pk, _ = find_peaks(amag[interior_idx], prominence=max(1e-8, 0.08 * grad_ref))
                if pk.size:
                    interior_peak = float(np.max(amag[interior_idx][pk]))
                    interior_ratio = interior_peak / max(min(abs(d0), abs(d2)), 1e-9)
                orientation_score = orientation_profile_score(
                    theta_prof[interior_idx], coh_prof[interior_idx], source_theta,
                    min_coherency=DENSE_EDGE_MIN_COHERENCY,
                    tolerance_deg=(26.0 if mode == 'thick' else 34.0),
                )

            if mode == 'thin' and interior_ratio > THIN_INTERIOR_PEAK_RATIO_MAX:
                continue
            if mode == 'thick' and orientation_score < THICK_MIN_ORIENTATION_PROFILE_SCORE:
                continue

            edge_score = np.clip(min(abs(d0), abs(d2)) / max(grad_ref, 1e-9), 0, 1)
            contrast_score = np.clip(contrast / max(2.0 * p['min_contrast'], 1e-9), 0, 1)
            bg_score = np.clip(1.0 - bg_difference / p['max_bg'], 0, 1)
            interior_soft = float(np.exp(-max(interior_ratio - 0.6, 0.0) / 1.2))
            coherency_score = float(np.clip(0.5 * (source_coh + opposite_coh), 0, 1))

            if mode == 'thin':
                interior_score = np.clip(1.0 - interior_ratio / THIN_INTERIOR_PEAK_RATIO_MAX, 0, 1)
                score = float(0.28 * edge_score + 0.30 * contrast_score +
                              0.20 * opposite_tangent_score + 0.12 * bg_score +
                              0.10 * interior_score)
            elif mode == 'thick':
                score = float(0.17 * edge_score + 0.13 * contrast_score +
                              0.16 * tensor_tangent_score + 0.16 * endpoint_theta_score +
                              0.25 * orientation_score + 0.06 * bg_score +
                              0.04 * interior_soft + 0.03 * coherency_score)
            else:  # curved: local geometry and along-curve continuity do most of the work
                score = float(0.22 * edge_score + 0.18 * contrast_score +
                              0.25 * tensor_tangent_score + 0.12 * endpoint_theta_score +
                              0.08 * orientation_score + 0.08 * bg_score +
                              0.04 * interior_soft + 0.03 * coherency_score)

            if score < p['min_score']:
                continue

            y1, x1 = float(y + ny * t0), float(x + nx * t0)
            candidates.append(dict(
                y1=y1, x1=x1, y2=y2, x2=x2,
                ym=0.5 * (y1 + y2), xm=0.5 * (x1 + x2),
                width_px=width, score=score, confidence=score,
                contrast=contrast, bg_difference=bg_difference,
                tangent_mismatch_deg=tangent_mismatch,
                endpoint_theta_diff_deg=endpoint_theta_diff,
                orientation_profile_score=orientation_score,
                interior_peak_ratio=interior_ratio,
                start_signed_gradient=d0, opposite_signed_gradient=d2,
                method=mode,
            ))

    candidates.sort(key=lambda d: d['score'], reverse=True)
    out = []
    for candidate in candidates:
        if any(abs(candidate['width_px'] - prior['width_px']) < 1.0 and
               np.hypot(candidate['y2'] - prior['y2'], candidate['x2'] - prior['x2']) < 1.5
               for prior in out):
            continue
        out.append(candidate)
        if len(out) >= p['max_candidates']:
            break
    return out


def viterbi_select(sample_records, mode):
    """Select a smooth opposite-edge path, allowing short evidence gaps in rescue modes."""
    chosen = [None] * len(sample_records)
    valid_indices = [i for i, record in enumerate(sample_records) if record['candidates']]
    if not valid_indices:
        return chosen

    max_gap = MAX_EMPTY_GAP_SAMPLES[mode]
    groups, group = [], [valid_indices[0]]
    for index in valid_indices[1:]:
        if index - group[-1] - 1 <= max_gap:
            group.append(index)
        else:
            groups.append(group)
            group = [index]
    groups.append(group)

    for indices in groups:
        costs, backs = [], []
        first_candidates = sample_records[indices[0]]['candidates']
        costs.append(np.array([1.0 - c['score'] for c in first_candidates], float))
        backs.append(np.full(len(first_candidates), -1, int))

        for j in range(1, len(indices)):
            previous_record = sample_records[indices[j - 1]]
            current_record = sample_records[indices[j]]
            prev_candidates = previous_record['candidates']
            cur_candidates = current_record['candidates']
            cur_cost = np.full(len(cur_candidates), np.inf)
            back = np.full(len(cur_candidates), -1, int)
            source_step = np.hypot(current_record['y'] - previous_record['y'],
                                   current_record['x'] - previous_record['x'])
            for k, candidate in enumerate(cur_candidates):
                local = 1.0 - candidate['score']
                for h, prior in enumerate(prev_candidates):
                    width_change = abs(np.log(max(candidate['width_px'], 1e-6) /
                                              max(prior['width_px'], 1e-6)))
                    opposite_step = np.hypot(candidate['y2'] - prior['y2'],
                                             candidate['x2'] - prior['x2'])
                    motion = abs(opposite_step - source_step) / max(source_step, 1.0)
                    value = (costs[j - 1][h] + local +
                             CONTINUITY_WEIGHT[mode] * width_change +
                             MOTION_WEIGHT[mode] * motion)
                    if value < cur_cost[k]:
                        cur_cost[k], back[k] = value, h
            costs.append(cur_cost)
            backs.append(back)

        state = int(np.argmin(costs[-1]))
        for j in range(len(indices) - 1, -1, -1):
            chosen[indices[j]] = sample_records[indices[j]]['candidates'][state]
            state = int(backs[j][state]) if j > 0 else -1
    return chosen


def robust_cv(values):
    v = np.asarray(values, float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return 1.4826 * mad / max(med, 1e-9)


def run_parameters(mode):
    if mode == 'thin':
        return THIN_MIN_RUN_SAMPLES, THIN_MIN_RUN_LENGTH_PX, THIN_MAX_RUN_ROBUST_CV, THIN_MIN_PAIR_SCORE
    if mode == 'thick':
        return THICK_MIN_RUN_SAMPLES, THICK_MIN_RUN_LENGTH_PX, THICK_MAX_RUN_ROBUST_CV, THICK_MIN_PAIR_SCORE
    if mode == 'curved':
        return CURVED_MIN_RUN_SAMPLES, CURVED_MIN_RUN_LENGTH_PX, CURVED_MAX_RUN_ROBUST_CV, CURVED_MIN_PAIR_SCORE
    raise ValueError(mode)


def measure_branch(img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
                   curves, jlab, mode):
    """Measure one branch: strict thin, straight thick rescue, or curved rescue."""
    H, W = img.shape
    min_width, max_width = branch_limits(mode, img.shape)
    if max_width <= min_width:
        return [], [], max_width

    search_edge = continuous_edge if mode == 'thin' else (continuous_edge | dense_edge)
    search_dist = ndimage.distance_transform_edt(~search_edge)
    avoid = np.zeros_like(continuous_edge, bool)
    if jlab.max() > 0:
        avoid |= dilation(jlab > 0, disk(JUNCTION_EXCLUSION_PX))
    avoid[:BORDER_EXCLUSION_PX] = True
    avoid[-BORDER_EXCLUSION_PX:] = True
    avoid[:, :BORDER_EXCLUSION_PX] = True
    avoid[:, -BORDER_EXCLUSION_PX:] = True

    grad_source = grad[continuous_edge]
    grad_ref = float(np.percentile(grad_source, 82)) if grad_source.size else float(np.percentile(grad, 95))
    grad_ref = max(grad_ref, 1e-8)
    min_samples, min_length, max_cv, min_pair_score = run_parameters(mode)

    accepted, rejected_runs = [], []
    run_counter = 0

    for contour_id, raw_path in enumerate(curves):
        path, arc = resample_polyline(raw_path, 1.0)
        if len(path) < MIN_EDGE_BRANCH_PX:
            continue
        smooth_sigma = max(1.0, min(3.0, 0.35 * THIN_MIN_WIDTH_PX + 0.8))
        sy = ndimage.gaussian_filter1d(path[:, 0], smooth_sigma, mode='nearest')
        sx = ndimage.gaussian_filter1d(path[:, 1], smooth_sigma, mode='nearest')
        ty_arr, tx_arr, directed_angle, local_turn_rate, good_tangent = local_turn_rate_and_angle(sy, sx, arc)

        sample_idx = np.arange(2, len(path) - 2, max(1, int(round(SAMPLE_SPACING_PX))))
        records = []
        for idx in sample_idx:
            y, x = float(sy[idx]), float(sx[idx])
            yi, xi = int(round(y)), int(round(x))
            permitted = (0 <= yi < H and 0 <= xi < W and not avoid[yi, xi] and good_tangent[idx])
            if mode == 'thick':
                permitted = permitted and local_turn_rate[idx] <= THICK_MAX_LOCAL_TURN_DEG_PER_PX
            elif mode == 'curved':
                permitted = permitted and local_turn_rate[idx] >= CURVED_MIN_LOCAL_TURN_DEG_PER_PX
            if not permitted:
                records.append(dict(index=int(idx), s=float(arc[idx]), y=y, x=x,
                                    tangent_angle=float(directed_angle[idx]),
                                    local_turn_rate=float(local_turn_rate[idx]), candidates=[]))
                continue

            ty, tx = float(ty_arr[idx]), float(tx_arr[idx])
            ny, nx = tx, -ty
            candidates = candidate_pairs_at_sample(
                img, gy, gx, grad, theta, coh, search_dist,
                y, x, ty, tx, ny, nx, min_width, max_width, grad_ref, mode)
            records.append(dict(index=int(idx), s=float(arc[idx]), y=y, x=x,
                                ty=ty, tx=tx,
                                tangent_angle=float(directed_angle[idx]),
                                local_turn_rate=float(local_turn_rate[idx]),
                                candidates=candidates))

        selected = viterbi_select(records, mode)
        blocks, current, trailing_gaps = [], [], 0
        for record, candidate in zip(records, selected):
            if candidate is None:
                if current:
                    trailing_gaps += 1
                    if trailing_gaps <= MAX_EMPTY_GAP_SAMPLES[mode]:
                        current.append(None)
                    else:
                        while current and current[-1] is None:
                            current.pop()
                        if current:
                            blocks.append(current)
                        current, trailing_gaps = [], 0
                continue

            trailing_gaps = 0
            item = dict(candidate, contour_id=contour_id,
                        sample_index=record['index'], arc_s=record['s'],
                        source_y=record['y'], source_x=record['x'],
                        source_tangent_angle=record['tangent_angle'],
                        local_turn_rate_deg_per_px=record['local_turn_rate'])
            valid_current = [d for d in current if d is not None]
            if valid_current:
                previous = valid_current[-1]
                jump = abs(item['width_px'] - previous['width_px']) / max(previous['width_px'], 1e-9)
                source_step = np.hypot(item['source_y'] - previous['source_y'],
                                       item['source_x'] - previous['source_x'])
                opposite_step = np.hypot(item['y2'] - previous['y2'], item['x2'] - previous['x2'])
                if jump > MAX_WIDTH_JUMP_FRAC or opposite_step > 3.5 * max(source_step, 1.0):
                    blocks.append(current)
                    current = []
            current.append(item)
        if current:
            blocks.append(current)

        for block in blocks:
            run_counter += 1
            clean_block = [d for d in block if d is not None]
            widths = np.array([d['width_px'] for d in clean_block], float)
            if not clean_block:
                continue
            run_length = float(clean_block[-1]['arc_s'] - clean_block[0]['arc_s']) if len(clean_block) > 1 else 0.0
            median_width = float(np.median(widths))
            mad = float(np.median(np.abs(widths - median_width)))
            keep = np.ones(len(clean_block), bool)
            if mad > 0:
                keep = np.abs(widths - median_width) <= 3.0 * 1.4826 * mad
            clean = [d for d, flag in zip(clean_block, keep) if flag]
            clean_widths = [d['width_px'] for d in clean]
            cv = robust_cv(clean_widths) if clean_widths else np.inf
            total_turn = 0.0
            if len(clean) > 1:
                angles = np.unwrap(np.array([d['source_tangent_angle'] for d in clean]))
                total_turn = float(np.degrees(np.max(angles) - np.min(angles)))

            reason = None
            if len(clean) < min_samples:
                reason = 'too_few_stable_samples'
            elif run_length < min_length:
                reason = 'paired_run_too_short'
            elif cv > max_cv:
                reason = 'width_not_stable'
            elif np.median([d['score'] for d in clean]) < min_pair_score:
                reason = 'pair_score_too_low'
            elif mode == 'curved' and total_turn < CURVED_MIN_RUN_TOTAL_TURN_DEG:
                reason = 'not_curved_enough'

            if reason:
                rejected_runs.append(dict(
                    method=mode, contour_id=contour_id, run_id=run_counter,
                    reason=reason, n_samples=len(clean), run_length_px=run_length,
                    robust_cv=cv, total_turn_deg=total_turn,
                    median_width_px=(float(np.median(clean_widths)) if clean_widths else np.nan),
                ))
                continue

            run_stability = float(np.clip(1.0 - cv / max(max_cv, 1e-6), 0, 1))
            run_length_score = float(np.clip(run_length / max(2.0 * min_length, 1e-6), 0, 1))
            for item in clean:
                item['run_id'] = run_counter
                item['run_length_px'] = run_length
                item['run_robust_cv'] = cv
                item['run_total_turn_deg'] = total_turn
                item['confidence'] = float(0.70 * item['score'] +
                                           0.20 * run_stability +
                                           0.10 * run_length_score)
                accepted.append(item)

    # Same chord may be found from both of its edges within one method.
    kept, keys = [], set()
    for item in sorted(accepted, key=lambda value: value['confidence'], reverse=True):
        a = (int(round(item['y1'] / DEDUP_ENDPOINT_BIN_PX)),
             int(round(item['x1'] / DEDUP_ENDPOINT_BIN_PX)))
        b = (int(round(item['y2'] / DEDUP_ENDPOINT_BIN_PX)),
             int(round(item['x2'] / DEDUP_ENDPOINT_BIN_PX)))
        key = tuple(sorted((a, b)))
        if key in keys:
            continue
        keys.add(key)
        kept.append(item)
    kept.sort(key=lambda value: (value['contour_id'], value['run_id'], value['arc_s']))
    return kept, rejected_runs, max_width


def chord_angle_deg(item):
    return float(np.degrees(np.arctan2(item['y2'] - item['y1'], item['x2'] - item['x1'])))


def are_duplicate_measurements(a, b):
    midpoint_distance = np.hypot(a['ym'] - b['ym'], a['xm'] - b['xm'])
    if midpoint_distance > MERGE_MIDPOINT_RADIUS_PX:
        return False
    if axial_angle_diff_deg(chord_angle_deg(a), chord_angle_deg(b)) > MERGE_MAX_CHORD_ANGLE_DEG:
        return False
    relative_width_difference = abs(a['width_px'] - b['width_px']) / max(min(a['width_px'], b['width_px']), 1e-9)
    return relative_width_difference <= MERGE_MAX_WIDTH_REL_DIFF


def prefer_measurement(a, b):
    """Choose between duplicates; strict thin wins when confidence is nearly tied."""
    confidence_gap = a['confidence'] - b['confidence']
    if abs(confidence_gap) <= 0.05:
        return a if METHOD_PRIORITY[a['method']] >= METHOD_PRIORITY[b['method']] else b
    return a if confidence_gap > 0 else b


def merge_method_results(thin_samples, thick_samples, curved_samples):
    merged = []
    candidates = list(thin_samples) + list(curved_samples) + list(thick_samples)
    candidates.sort(key=lambda item: (item['confidence'], METHOD_PRIORITY[item['method']]), reverse=True)
    for candidate in candidates:
        duplicate_index = next((i for i, existing in enumerate(merged)
                                if are_duplicate_measurements(candidate, existing)), None)
        if duplicate_index is None:
            merged.append(candidate)
        else:
            merged[duplicate_index] = prefer_measurement(candidate, merged[duplicate_index])
    merged.sort(key=lambda item: (item.get('contour_id', -1), item.get('arc_s', 0.0)))
    return merged


# ------------------------------------------------------------- output + visualization
def add_chords(ax, samples, color=None, cmap=None, linewidth=1.2, label=None):
    if not samples:
        return None
    segments = [[(d['x1'], d['y1']), (d['x2'], d['y2'])] for d in samples]
    if cmap is not None:
        widths = np.array([d['width_px'] for d in samples], float)
        lo, hi = np.percentile(widths, [5, 95]) if len(widths) > 1 else (widths[0] - 1, widths[0] + 1)
        collection = LineCollection(segments, cmap=cmap,
                                    norm=plt.Normalize(lo, hi if hi > lo else lo + 1),
                                    linewidths=linewidth, alpha=0.90)
        collection.set_array(widths)
    else:
        collection = LineCollection(segments, colors=color, linewidths=linewidth,
                                    alpha=0.90, label=label)
    ax.add_collection(collection)
    return collection


def draw_result(img, continuous_edge, dense_edge, curves,
                thin_samples, thick_samples, curved_samples, merged_samples,
                name, output_png, max_width):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax in axes.ravel():
        ax.imshow(img, cmap='gray')
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis('off')

    axes[0, 0].imshow(np.ma.masked_where(~continuous_edge, continuous_edge), cmap='cool', alpha=0.90)
    axes[0, 0].set_title(
        f'1. traced Canny edges\npixels={int(continuous_edge.sum())}, curves={len(curves)}')

    axes[0, 1].imshow(np.ma.masked_where(~dense_edge, dense_edge), cmap='autumn', alpha=0.75)
    axes[0, 1].set_title(f'2. dense opposite-edge evidence\npixels={int(dense_edge.sum())}')

    add_chords(axes[0, 2], thin_samples, color='cyan', linewidth=1.4, label='thin')
    add_chords(axes[0, 2], thick_samples, color='orange', linewidth=1.4, label='thick rescue')
    add_chords(axes[0, 2], curved_samples, color='lime', linewidth=1.4, label='curved rescue')
    axes[0, 2].legend(loc='upper right', fontsize=8, framealpha=0.75)
    axes[0, 2].set_title(
        f'3. raw branch samples before merge\nthin={len(thin_samples)}, thick={len(thick_samples)}, curved={len(curved_samples)}')

    collection = add_chords(axes[1, 0], merged_samples, cmap=plt.cm.plasma, linewidth=1.35)
    if collection is not None:
        cb = fig.colorbar(collection, ax=axes[1, 0], fraction=0.046, pad=0.02)
        cb.set_label('thickness (px)')
        median = np.median([d['width_px'] for d in merged_samples])
        axes[1, 0].set_title(f'4. merged local-normal chords\nmedian={median:.2f} px, n={len(merged_samples)}')
    else:
        axes[1, 0].set_title('4. no measurement survived')

    axes[1, 1].clear()
    method_colors = {'thin': 'tab:blue', 'thick': 'tab:orange', 'curved': 'tab:green'}
    for method in ('thin', 'thick', 'curved'):
        values = [d['width_px'] for d in merged_samples if d['method'] == method]
        if values:
            axes[1, 1].hist(values, bins='auto', alpha=0.55,
                            color=method_colors[method], label=f'{method} (n={len(values)})')
    axes[1, 1].set_xlabel('thickness (px)')
    axes[1, 1].set_ylabel('merged local measurements')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title('5. distribution by accepted mechanism')
    axes[1, 1].set_aspect('auto')
    axes[1, 1].axis('on')

    axes[1, 2].clear()
    if merged_samples:
        widths = np.array([d['width_px'] for d in merged_samples], float)
        confidence = np.array([d['confidence'] for d in merged_samples], float)
        for method in ('thin', 'thick', 'curved'):
            mask = np.array([d['method'] == method for d in merged_samples])
            if mask.any():
                axes[1, 2].scatter(widths[mask], confidence[mask], s=13, alpha=0.65,
                                   color=method_colors[method], label=method)
        axes[1, 2].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
        axes[1, 2].set_xlabel('thickness (px)')
        axes[1, 2].set_ylabel('confidence')
        axes[1, 2].set_ylim(0, 1.02)
        axes[1, 2].legend(fontsize=8)
    else:
        axes[1, 2].text(0.5, 0.5, 'No accepted measurements', ha='center', va='center')
    axes[1, 2].set_title('6. thickness vs confidence')
    axes[1, 2].set_aspect('auto')
    axes[1, 2].axis('on')

    fig.suptitle(f'{name} | hybrid thin + thick + curved | max width={max_width:g}px', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches='tight')
    if SHOW_INLINE:
        plt.show()
    plt.close(fig)


def process_one(path):
    img, cut = prepare_image(path)
    gy, gx, grad, theta, coh, energy = orientation_fields(img)
    continuous_edge = build_continuous_edges(img, coh, energy)
    dense_edge, dense_threshold = build_dense_edge_candidates(grad, gy, gx, coh, energy)
    curves, jlab = trace_edge_curves(continuous_edge)

    thin_samples, thin_rejected, thin_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, curves, jlab, 'thin')
    thick_samples, thick_rejected, thick_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, curves, jlab, 'thick')
    curved_samples, curved_rejected, curved_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, curves, jlab, 'curved')
    merged_samples = merge_method_results(thin_samples, thick_samples, curved_samples)

    sample_df = pd.DataFrame(merged_samples)
    if len(sample_df):
        sample_df.insert(0, 'file', path.name)
        sample_df['status'] = 'accepted'
        nm_px = NM_PER_PX.get(path.name)
        sample_df['nm_per_px'] = nm_px if nm_px is not None else np.nan
        sample_df['thickness_nm'] = sample_df['width_px'] * nm_px if nm_px is not None else np.nan

    branch_df = pd.DataFrame(thin_samples + thick_samples + curved_samples)
    if len(branch_df):
        branch_df.insert(0, 'file', path.name)

    reject_df = pd.DataFrame(thin_rejected + thick_rejected + curved_rejected)
    if len(reject_df):
        reject_df.insert(0, 'file', path.name)

    output_png = OUTPUT_DIR / f'{path.stem}_hybrid_edge_thickness.png'
    draw_result(img, continuous_edge, dense_edge, curves,
                thin_samples, thick_samples, curved_samples, merged_samples,
                path.name, output_png, max(thin_max, thick_max, curved_max))
    sample_df.to_csv(OUTPUT_DIR / f'{path.stem}_merged_thickness_samples.csv', index=False)
    branch_df.to_csv(OUTPUT_DIR / f'{path.stem}_all_branch_candidates.csv', index=False)
    reject_df.to_csv(OUTPUT_DIR / f'{path.stem}_rejected_runs.csv', index=False)

    widths = sample_df['width_px'].to_numpy(float) if len(sample_df) else np.array([])
    nm_px = NM_PER_PX.get(path.name)
    method_counts = sample_df['method'].value_counts().to_dict() if len(sample_df) else {}
    summary = dict(
        file=path.name, nm_per_px=(nm_px if nm_px is not None else np.nan),
        cropped_height=cut, image_height=img.shape[0], image_width=img.shape[1],
        continuous_edge_pixels=int(continuous_edge.sum()),
        dense_edge_pixels=int(dense_edge.sum()), dense_gradient_threshold=dense_threshold,
        traced_edge_curves=len(curves),
        thin_measurements=int(method_counts.get('thin', 0)),
        thick_measurements=int(method_counts.get('thick', 0)),
        curved_measurements=int(method_counts.get('curved', 0)),
        accepted_measurements=len(widths),
        accepted_runs=(sample_df[['method', 'run_id']].drop_duplicates().shape[0] if len(sample_df) else 0),
        rejected_runs=len(reject_df),
        max_width_px_used=max(thin_max, thick_max, curved_max),
        median_thickness_px=(float(np.median(widths)) if len(widths) else np.nan),
        p25_thickness_px=(float(np.percentile(widths, 25)) if len(widths) else np.nan),
        p75_thickness_px=(float(np.percentile(widths, 75)) if len(widths) else np.nan),
        mean_thickness_px=(float(np.mean(widths)) if len(widths) else np.nan),
        median_thickness_nm=(float(np.median(widths)) * nm_px
                             if len(widths) and nm_px is not None else np.nan),
    )
    return summary, sample_df


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f'Folder not found: {IMAGE_DIR}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = []
    missing = []
    for name in TARGET_FILES:
        path = IMAGE_DIR / name
        if path.is_file():
            files.append(path)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError('Requested image(s) not found: ' + ', '.join(missing))

    print(f'Input : {IMAGE_DIR}')
    print(f'Output: {OUTPUT_DIR}')
    print('Images: ' + ', '.join(path.name for path in files))

    summaries, all_samples = [], []
    for index, path in enumerate(files, 1):
        try:
            print(f'[{index}/{len(files)}] {path.name}')
            summary, frame = process_one(path)
            summaries.append(summary)
            if len(frame):
                all_samples.append(frame)
            median = summary['median_thickness_px']
            if np.isfinite(median):
                print(f"    thin={summary['thin_measurements']}, "
                      f"thick={summary['thick_measurements']}, "
                      f"curved={summary['curved_measurements']}, "
                      f"merged={summary['accepted_measurements']}, median={median:.2f}px")
            else:
                print('    accepted=0')
        except Exception as exc:
            warnings.warn(f'{path.name}: {type(exc).__name__}: {exc}')
            summaries.append(dict(file=path.name, error=f'{type(exc).__name__}: {exc}'))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / 'hybrid_thickness_summary.csv', index=False)
    if all_samples:
        pd.concat(all_samples, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_merged_thickness_samples.csv', index=False)

    print('\nDone.')
    print(f'Saved diagnostics and CSV files under:\n{OUTPUT_DIR}')
    display(summary_df) if 'display' in globals() else print(summary_df)


# ============================================================================
# Consensus + adaptive ribbon-block extension
# - Detector 1: bright multi-scale ridge centerline + bilateral edge search
# - Detector 2: existing continuous-edge / local-normal pipeline above
# - Grade A: both detectors agree
# - Grade B: one detector is extended by a long high-confidence block track
# - Grade C: single-detector or weak/short block candidates (visualization only)
# ============================================================================
from scipy.spatial import cKDTree
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects


def _remove_small_objects_compat(mask, min_pixels):
    try:
        return remove_small_objects(mask, max_size=max(0, int(min_pixels) - 1))
    except TypeError:
        return remove_small_objects(mask, min_size=int(min_pixels))

OUTPUT_DIR = IMAGE_DIR.parent / f'{IMAGE_DIR.name}_edge_thickness_consensus_block'

# Bright-ridge centerline detector.
RIDGE_SIGMAS = (1.2, 1.8, 2.6, 3.6, 5.0, 7.0, 9.0)
RIDGE_RESPONSE_PERCENTILE = 80.0
RIDGE_MIN_INTENSITY_PERCENTILE = 35.0
RIDGE_CLOSE_RADIUS_PX = 1
RIDGE_MIN_COMPONENT_PX = 20
RIDGE_SAMPLE_SPACING_PX = 3.0
RIDGE_MIN_WIDTH_PX = 3.0
RIDGE_MAX_WIDTH_PX = None              # automatic: min(80 px, 16% of short image side)
RIDGE_SIDE_MAX_CANDIDATES = 5
RIDGE_EDGE_TOLERANCE_PX = 2.1
RIDGE_MIN_PAIR_SCORE = 0.39
RIDGE_MIN_RUN_SAMPLES = 5
RIDGE_MIN_RUN_LENGTH_PX = 15.0
RIDGE_MAX_RUN_ROBUST_CV = 0.36
RIDGE_ALLOWED_GAP_SAMPLES = 1

# Detector consensus.
CONSENSUS_RADIUS_PX = 4.5
CONSENSUS_MAX_CHORD_ANGLE_DEG = 20.0
CONSENSUS_MAX_TANGENT_DIFF_DEG = 25.0
CONSENSUS_MAX_WIDTH_REL_DIFF = 0.38
CONSENSUS_MIN_SCORE = 0.50

# Adaptive ribbon-block tracking.
BLOCK_TRIGGER_WIDTH_PX = 10.0
BLOCK_MIN_ANCHOR_POINTS = 3
BLOCK_MIN_ANCHOR_SPAN_PX = 6.0
BLOCK_STRONG_CANDIDATE_CONFIDENCE = 0.60
BLOCK_MAX_ANCHORS = 36
BLOCK_ANCHOR_NMS_RADIUS_PX = 12.0
BLOCK_STEP_PX = 3.0
BLOCK_TURN_DEGREES = (0, -5, 5, -10, 10, -15, 15, -20, 20,
                      -25, 25, -30, 30, -35, 35, -40, 40, -45, 45)
BLOCK_ANGLE_BRANCHES = 5
BLOCK_BEAM_WIDTH = 8
BLOCK_WIDTH_DELTA_PX = 0.85
BLOCK_MIN_HALF_WIDTH_PX = 1.25
BLOCK_MAX_STEPS = 110
BLOCK_MAX_WEAK_STEPS = 4
BLOCK_WEAK_LOCAL_SCORE = 0.40
BLOCK_MIN_TRACK_STEPS = 8
BLOCK_MIN_TRACK_LENGTH_PX = 24.0
BLOCK_GRADE_B_MIN_AVG_CONFIDENCE = 0.54
BLOCK_GRADE_B_MIN_EDGE_SUPPORT = 0.40
BLOCK_GRADE_B_MIN_ORIENTATION = 0.45
BLOCK_PATH_SAMPLE_STRIDE = 1
BLOCK_LOOP_RADIUS_PX = 2.5
BLOCK_JUNCTION_RADIUS_PX = 4.0

GRADE_PRIORITY = {'A': 3, 'B': 2, 'C': 1}


def tangent_angle_from_item(item):
    if 'tangent_angle_rad' in item and np.isfinite(item['tangent_angle_rad']):
        return float(item['tangent_angle_rad'])
    if 'source_tangent_angle' in item and np.isfinite(item['source_tangent_angle']):
        return float(item['source_tangent_angle'])
    if all(k in item for k in ('center_y', 'center_x', 'y1', 'x1')):
        # Chord is normal; rotate it by 90 degrees to obtain the tangent.
        return float(np.arctan2(item['y2'] - item['y1'], item['x2'] - item['x1']) + np.pi / 2)
    return 0.0


def canonicalize_measurement(item, detector=None):
    d = dict(item)
    d['center_y'] = float(d.get('center_y', d.get('ym', 0.5 * (d['y1'] + d['y2']))))
    d['center_x'] = float(d.get('center_x', d.get('xm', 0.5 * (d['x1'] + d['x2']))))
    d['ym'], d['xm'] = d['center_y'], d['center_x']
    d['width_px'] = float(d.get('width_px', np.hypot(d['y2'] - d['y1'], d['x2'] - d['x1'])))
    d['tangent_angle_rad'] = tangent_angle_from_item(d)
    d['tangent_angle_deg'] = float(np.degrees(d['tangent_angle_rad']) % 180.0)
    d['confidence'] = float(d.get('confidence', d.get('score', 0.0)))
    d['score'] = float(d.get('score', d['confidence']))
    if detector is not None:
        d['detector'] = detector
    return d


def build_bright_ridge_centerlines(img):
    """Find continuous bright ridges; these are centerline hypotheses, not final fibres."""
    response = sato(img, sigmas=RIDGE_SIGMAS, black_ridges=False, mode='reflect')
    response = np.asarray(response, np.float32)
    finite = np.isfinite(response)
    if not finite.any() or float(np.max(response[finite])) <= 0:
        return np.zeros_like(img), np.zeros_like(img, bool), [], np.zeros_like(img, np.int32)
    lo, hi = np.percentile(response[finite], [5, 99.7])
    response_n = np.clip((response - lo) / max(float(hi - lo), 1e-9), 0, 1).astype(np.float32)
    rthr = float(np.percentile(response_n[finite], RIDGE_RESPONSE_PERCENTILE))
    ithr = float(np.percentile(img[np.isfinite(img)], RIDGE_MIN_INTENSITY_PERCENTILE))
    mask = finite & (response_n >= rthr) & (img >= ithr)
    if RIDGE_CLOSE_RADIUS_PX > 0:
        mask = closing(mask, disk(RIDGE_CLOSE_RADIUS_PX))
    mask = _remove_small_objects_compat(mask.astype(bool), RIDGE_MIN_COMPONENT_PX)
    centerline = skeletonize(mask)
    centerline = remove_small_components(centerline, RIDGE_MIN_COMPONENT_PX)
    curves, junction_labels = trace_edge_curves(centerline)
    return response_n, centerline.astype(bool), curves, junction_labels


def _sample_line(a, y, x, dy, dx, ts):
    return ndimage.map_coordinates(
        a, np.vstack([y + dy * ts, x + dx * ts]), order=1, mode='nearest')


def _ridge_side_candidates(img, gy, gx, grad, theta, coh, edge_dist,
                           y, x, ty, tx, outward_y, outward_x,
                           max_half_width, grad_ref):
    ts = np.arange(1.0, max_half_width + PROFILE_OUTSIDE_PX + 2.0,
                   PROFILE_STEP_PX, dtype=np.float32)
    yy, xx = y + outward_y * ts, x + outward_x * ts
    valid = ((yy >= 1) & (yy < img.shape[0] - 1) &
             (xx >= 1) & (xx < img.shape[1] - 1))
    if valid.sum() < 8:
        return []
    prof = _sample_line(img, y, x, outward_y, outward_x, ts)
    py = _sample_line(gy, y, x, outward_y, outward_x, ts)
    px = _sample_line(gx, y, x, outward_y, outward_x, ts)
    amag = np.abs(py * outward_y + px * outward_x)
    signed = py * outward_y + px * outward_x
    peaks, props = find_peaks(
        amag,
        distance=max(1, int(round(1.3 / PROFILE_STEP_PX))),
        prominence=max(1e-8, 0.065 * grad_ref),
    )
    center_level = float(np.median(_sample_line(
        img, y, x, outward_y, outward_x, np.array([0.0, 0.7, 1.4], np.float32))))
    out = []
    for ip in peaks:
        half_width = float(ts[ip])
        if half_width < BLOCK_MIN_HALF_WIDTH_PX or half_width > max_half_width:
            continue
        by, bx = float(yy[ip]), float(xx[ip])
        distance = sample_map(edge_dist, by, bx)
        if distance > RIDGE_EDGE_TOLERANCE_PX:
            continue
        boundary_theta = sample_map(theta, by, bx)
        tangent_mismatch = axial_tangent_mismatch(boundary_theta, ty, tx)
        if tangent_mismatch > 42.0:
            continue
        outside_mask = (ts >= half_width + 0.8) & (ts <= half_width + PROFILE_OUTSIDE_PX)
        if outside_mask.sum() < 2:
            continue
        outside_level = float(np.median(prof[outside_mask]))
        contrast = center_level - outside_level
        if contrast < 0.018:
            continue
        edge_strength = float(np.clip(amag[ip] / max(grad_ref, 1e-9), 0, 1))
        support = float(np.exp(-0.5 * (distance / max(RIDGE_EDGE_TOLERANCE_PX, 1e-6)) ** 2))
        tangent_score = float(np.exp(-((tangent_mismatch / 28.0) ** 2)))
        contrast_score = float(np.clip(contrast / 0.11, 0, 1))
        # Moving from a bright center to the outside should usually be descending.
        polarity_score = 1.0 if signed[ip] < 0 else 0.35
        score = (0.28 * edge_strength + 0.25 * support + 0.20 * tangent_score +
                 0.20 * contrast_score + 0.07 * polarity_score)
        out.append(dict(
            half_width_px=half_width, y=by, x=bx, score=float(score),
            edge_strength=edge_strength, edge_support=support,
            tangent_score=tangent_score, contrast=contrast,
            signed_gradient=float(signed[ip]), boundary_theta=float(boundary_theta),
        ))
    out.sort(key=lambda d: d['score'], reverse=True)
    return out[:RIDGE_SIDE_MAX_CANDIDATES]


def ridge_pairs_at_sample(img, ridge_response, gy, gx, grad, theta, coh, edge_dist,
                          y, x, ty, tx, min_width, max_width, grad_ref):
    ny, nx = tx, -ty
    max_half = min(max_width, max_width * 0.80)
    left = _ridge_side_candidates(img, gy, gx, grad, theta, coh, edge_dist,
                                  y, x, ty, tx, -ny, -nx, max_half, grad_ref)
    right = _ridge_side_candidates(img, gy, gx, grad, theta, coh, edge_dist,
                                   y, x, ty, tx, ny, nx, max_half, grad_ref)
    if not left or not right:
        return []
    center_theta = sample_map(theta, y, x)
    center_coh = sample_map(coh, y, x)
    center_tangent = float(np.clip(1.0 - axial_tangent_mismatch(center_theta, ty, tx) / 45.0, 0, 1))
    ridge_score = sample_map(ridge_response, y, x)
    candidates = []
    for L in left:
        for R in right:
            width = L['half_width_px'] + R['half_width_px']
            if not (min_width <= width <= max_width):
                continue
            # Check orientation inside the whole ribbon. Direction may be noisy locally,
            # so coherency is used as a weight rather than a hard gate.
            ts = np.linspace(-L['half_width_px'] + 1.0,
                             R['half_width_px'] - 1.0,
                             max(5, int(round(width / 2.0))))
            theta_inside = _sample_line(theta, y, x, ny, nx, ts)
            coh_inside = _sample_line(coh, y, x, ny, nx, ts)
            orientation_score = orientation_profile_score(
                theta_inside, coh_inside, center_theta,
                min_coherency=0.08, tolerance_deg=34.0)
            asymmetry = abs(L['half_width_px'] - R['half_width_px']) / max(width, 1e-9)
            asymmetry_score = float(np.exp(-((asymmetry / 0.42) ** 2)))
            boundary_score = 0.5 * (L['score'] + R['score'])
            pair_score = float(
                0.35 * boundary_score + 0.20 * ridge_score +
                0.16 * orientation_score + 0.12 * center_tangent +
                0.10 * np.clip(center_coh, 0, 1) + 0.07 * asymmetry_score)
            if pair_score < RIDGE_MIN_PAIR_SCORE:
                continue
            y1, x1 = L['y'], L['x']
            y2, x2 = R['y'], R['x']
            candidates.append(dict(
                y1=y1, x1=x1, y2=y2, x2=x2,
                ym=float(y), xm=float(x), center_y=float(y), center_x=float(x),
                left_width_px=float(L['half_width_px']),
                right_width_px=float(R['half_width_px']),
                width_px=float(width), score=pair_score, confidence=pair_score,
                ridge_response=float(ridge_score),
                orientation_profile_score=float(orientation_score),
                center_tangent_score=float(center_tangent),
                left_edge_support=float(L['edge_support']),
                right_edge_support=float(R['edge_support']),
                edge_support=float(0.5 * (L['edge_support'] + R['edge_support'])),
                method='ridge', detector='ridge',
            ))
    candidates.sort(key=lambda d: d['score'], reverse=True)
    return candidates[:8]


def viterbi_select_ridge(records):
    selected = [None] * len(records)
    valid = [i for i, r in enumerate(records) if r['candidates']]
    if not valid:
        return selected
    groups, group = [], [valid[0]]
    for idx in valid[1:]:
        if idx - group[-1] - 1 <= RIDGE_ALLOWED_GAP_SAMPLES:
            group.append(idx)
        else:
            groups.append(group); group = [idx]
    groups.append(group)
    for indices in groups:
        first = records[indices[0]]['candidates']
        costs = [np.array([1.0 - c['score'] for c in first])]
        backs = [np.full(len(first), -1, int)]
        for j in range(1, len(indices)):
            prev = records[indices[j - 1]]['candidates']
            cur = records[indices[j]]['candidates']
            cst = np.full(len(cur), np.inf); back = np.full(len(cur), -1, int)
            source_step = np.hypot(records[indices[j]]['y'] - records[indices[j - 1]]['y'],
                                   records[indices[j]]['x'] - records[indices[j - 1]]['x'])
            for k, cand in enumerate(cur):
                for h, prior in enumerate(prev):
                    dl = abs(cand['left_width_px'] - prior['left_width_px']) / max(prior['left_width_px'], 1.0)
                    dr = abs(cand['right_width_px'] - prior['right_width_px']) / max(prior['right_width_px'], 1.0)
                    left_motion = abs(np.hypot(cand['y1'] - prior['y1'], cand['x1'] - prior['x1']) - source_step)
                    right_motion = abs(np.hypot(cand['y2'] - prior['y2'], cand['x2'] - prior['x2']) - source_step)
                    value = costs[j - 1][h] + (1.0 - cand['score']) + 1.45 * (dl + dr) + 0.18 * (left_motion + right_motion)
                    if value < cst[k]:
                        cst[k], back[k] = value, h
            costs.append(cst); backs.append(back)
        state = int(np.argmin(costs[-1]))
        for j in range(len(indices) - 1, -1, -1):
            selected[indices[j]] = records[indices[j]]['candidates'][state]
            state = int(backs[j][state]) if j > 0 else -1
    return selected


def measure_ridge_detector(img, ridge_response, centerline_curves,
                           gy, gx, grad, theta, coh, continuous_edge, dense_edge):
    search_edge = continuous_edge | dense_edge
    edge_dist = ndimage.distance_transform_edt(~search_edge)
    max_width = min(80.0, 0.16 * min(img.shape)) if RIDGE_MAX_WIDTH_PX is None else float(RIDGE_MAX_WIDTH_PX)
    min_width = RIDGE_MIN_WIDTH_PX
    grad_values = grad[search_edge]
    grad_ref = float(np.percentile(grad_values, 80)) if grad_values.size else float(np.percentile(grad, 94))
    grad_ref = max(grad_ref, 1e-8)
    accepted, rejected = [], []
    run_id = 0
    for contour_id, raw in enumerate(centerline_curves):
        path, arc = resample_polyline(raw, 1.0)
        if len(path) < RIDGE_MIN_COMPONENT_PX:
            continue
        sy = ndimage.gaussian_filter1d(path[:, 0], 2.0, mode='nearest')
        sx = ndimage.gaussian_filter1d(path[:, 1], 2.0, mode='nearest')
        ty, tx, directed, turn_rate, good = local_turn_rate_and_angle(sy, sx, arc)
        records = []
        for idx in np.arange(2, len(path) - 2, max(1, int(round(RIDGE_SAMPLE_SPACING_PX)))):
            y, x = float(sy[idx]), float(sx[idx])
            if not good[idx] or not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
                candidates = []
            else:
                candidates = ridge_pairs_at_sample(
                    img, ridge_response, gy, gx, grad, theta, coh, edge_dist,
                    y, x, float(ty[idx]), float(tx[idx]), min_width, max_width, grad_ref)
            records.append(dict(index=int(idx), arc_s=float(arc[idx]), y=y, x=x,
                                ty=float(ty[idx]), tx=float(tx[idx]),
                                tangent_angle_rad=float(directed[idx]),
                                local_turn_rate_deg_per_px=float(turn_rate[idx]),
                                candidates=candidates))
        chosen = viterbi_select_ridge(records)
        runs, current, gap = [], [], 0
        for record, candidate in zip(records, chosen):
            if candidate is None:
                if current:
                    gap += 1
                    if gap <= RIDGE_ALLOWED_GAP_SAMPLES:
                        current.append(None)
                    else:
                        while current and current[-1] is None: current.pop()
                        if current: runs.append(current)
                        current, gap = [], 0
                continue
            gap = 0
            item = dict(candidate,
                        contour_id=contour_id, arc_s=record['arc_s'],
                        tangent_angle_rad=record['tangent_angle_rad'],
                        tangent_angle_deg=float(np.degrees(record['tangent_angle_rad']) % 180),
                        local_turn_rate_deg_per_px=record['local_turn_rate_deg_per_px'])
            valid_current = [v for v in current if v is not None]
            if valid_current:
                prior = valid_current[-1]
                width_jump = abs(item['width_px'] - prior['width_px']) / max(prior['width_px'], 1e-9)
                center_jump = np.hypot(item['center_y'] - prior['center_y'], item['center_x'] - prior['center_x'])
                if width_jump > 0.52 or center_jump > 3.5 * RIDGE_SAMPLE_SPACING_PX:
                    runs.append(current); current = []
            current.append(item)
        if current: runs.append(current)
        for run in runs:
            run_id += 1
            clean = [v for v in run if v is not None]
            if not clean:
                continue
            length = clean[-1]['arc_s'] - clean[0]['arc_s'] if len(clean) > 1 else 0.0
            widths = [v['width_px'] for v in clean]
            cv = robust_cv(widths)
            median_score = float(np.median([v['score'] for v in clean]))
            reason = None
            if len(clean) < RIDGE_MIN_RUN_SAMPLES: reason = 'too_few_ridge_samples'
            elif length < RIDGE_MIN_RUN_LENGTH_PX: reason = 'ridge_run_too_short'
            elif cv > RIDGE_MAX_RUN_ROBUST_CV: reason = 'ridge_width_not_stable'
            elif median_score < RIDGE_MIN_PAIR_SCORE: reason = 'ridge_score_too_low'
            if reason:
                rejected.append(dict(detector='ridge', contour_id=contour_id, run_id=run_id,
                                     reason=reason, n_samples=len(clean), run_length_px=length,
                                     robust_cv=cv, median_score=median_score))
                continue
            run_stability = np.clip(1.0 - cv / RIDGE_MAX_RUN_ROBUST_CV, 0, 1)
            length_score = np.clip(length / (2.0 * RIDGE_MIN_RUN_LENGTH_PX), 0, 1)
            for item in clean:
                item['run_id'] = run_id
                item['run_length_px'] = float(length)
                item['run_robust_cv'] = float(cv)
                item['confidence'] = float(0.72 * item['score'] + 0.18 * run_stability + 0.10 * length_score)
                item['grade'] = 'C'
                accepted.append(canonicalize_measurement(item, 'ridge'))
    return accepted, rejected, max_width


def _measurement_match_score(ridge, edge):
    center_dist = np.hypot(ridge['center_y'] - edge['center_y'], ridge['center_x'] - edge['center_x'])
    if center_dist > CONSENSUS_RADIUS_PX:
        return None
    chord_diff = axial_angle_diff_deg(chord_angle_deg(ridge), chord_angle_deg(edge))
    if chord_diff > CONSENSUS_MAX_CHORD_ANGLE_DEG:
        return None
    tangent_diff = axial_angle_diff_deg(np.degrees(tangent_angle_from_item(ridge)),
                                        np.degrees(tangent_angle_from_item(edge)))
    if tangent_diff > CONSENSUS_MAX_TANGENT_DIFF_DEG:
        return None
    width_rel = abs(ridge['width_px'] - edge['width_px']) / max(min(ridge['width_px'], edge['width_px']), 1e-9)
    if width_rel > CONSENSUS_MAX_WIDTH_REL_DIFF:
        return None
    geometry = (np.exp(-0.5 * (center_dist / CONSENSUS_RADIUS_PX) ** 2) +
                np.exp(-0.5 * (chord_diff / CONSENSUS_MAX_CHORD_ANGLE_DEG) ** 2) +
                np.exp(-0.5 * (tangent_diff / CONSENSUS_MAX_TANGENT_DIFF_DEG) ** 2) +
                np.exp(-0.5 * (width_rel / CONSENSUS_MAX_WIDTH_REL_DIFF) ** 2)) / 4.0
    return float(0.65 * geometry + 0.175 * ridge['confidence'] + 0.175 * edge['confidence'])


def match_detector_consensus(ridge_samples, edge_samples):
    ridge = [canonicalize_measurement(v, 'ridge') for v in ridge_samples]
    edge = [canonicalize_measurement(v, 'edge') for v in edge_samples]
    if not ridge or not edge:
        for d in ridge + edge: d['grade'] = 'C'
        return [], ridge, edge
    tree = cKDTree(np.array([[d['center_y'], d['center_x']] for d in edge], float))
    proposals = []
    for i, r in enumerate(ridge):
        for j in tree.query_ball_point([r['center_y'], r['center_x']], CONSENSUS_RADIUS_PX):
            score = _measurement_match_score(r, edge[j])
            if score is not None and score >= CONSENSUS_MIN_SCORE:
                proposals.append((score, i, j))
    proposals.sort(reverse=True)
    used_r, used_e, confirmed = set(), set(), []
    for agreement, i, j in proposals:
        if i in used_r or j in used_e:
            continue
        r, e = ridge[i], edge[j]
        used_r.add(i); used_e.add(j)
        # Keep the ridge center and left/right widths; average total geometry with edge evidence.
        item = dict(r)
        item['width_px'] = float(0.55 * r['width_px'] + 0.45 * e['width_px'])
        scale = item['width_px'] / max(r['width_px'], 1e-9)
        cy, cx = r['center_y'], r['center_x']
        item['y1'] = cy + scale * (r['y1'] - cy); item['x1'] = cx + scale * (r['x1'] - cx)
        item['y2'] = cy + scale * (r['y2'] - cy); item['x2'] = cx + scale * (r['x2'] - cx)
        item['left_width_px'] = float(r.get('left_width_px', 0.5 * item['width_px']) * scale)
        item['right_width_px'] = float(r.get('right_width_px', 0.5 * item['width_px']) * scale)
        item['method'] = 'consensus'; item['detector'] = 'ridge+edge'; item['grade'] = 'A'
        item['detector_agreement_score'] = float(agreement)
        item['confidence'] = float(np.clip(0.45 * r['confidence'] + 0.35 * e['confidence'] + 0.20 * agreement + 0.08, 0, 1))
        item['ridge_contour_id'] = int(r.get('contour_id', -1)); item['ridge_run_id'] = int(r.get('run_id', -1))
        item['ridge_arc_s'] = float(r.get('arc_s', 0.0))
        item['edge_contour_id'] = int(e.get('contour_id', -1)); item['edge_run_id'] = int(e.get('run_id', -1))
        item['edge_method'] = e.get('method', 'edge')
        confirmed.append(item)
    ridge_only = [dict(v, grade='C') for i, v in enumerate(ridge) if i not in used_r]
    edge_only = [dict(v, grade='C') for j, v in enumerate(edge) if j not in used_e]
    return confirmed, ridge_only, edge_only


def propose_half_widths(width, trend, min_width, max_width):
    proposals = []
    for perturb in (-BLOCK_WIDTH_DELTA_PX, 0.0, BLOCK_WIDTH_DELTA_PX):
        delta = float(trend + perturb)
        new_width = float(np.clip(width + delta, min_width, max_width))
        actual = new_width - width
        new_trend = float(0.68 * trend + 0.32 * actual)
        key = round(new_width, 3)
        if not any(round(p[0], 3) == key for p in proposals):
            proposals.append((new_width, new_trend, actual))
    return proposals


def representative_width_combos(left_options, right_options):
    if not left_options or not right_options:
        return []
    def triplet(options):
        low = options[0]
        high = options[-1]
        mid = min(options, key=lambda v: abs(v[2]))
        return low, mid, high
    l0, lm, l1 = triplet(left_options)
    r0, rm, r1 = triplet(right_options)
    raw = [(lm, rm), (l0, r0), (l1, r1), (l0, rm), (l1, rm), (lm, r0), (lm, r1)]
    out, seen = [], set()
    for left, right in raw:
        key = (round(left[0], 3), round(right[0], 3))
        if key not in seen:
            seen.add(key); out.append((left, right))
    return out


def _anchor_from_sequence(seq, at_end, source):
    k = min(5, len(seq))
    recent = seq[-k:] if at_end else list(reversed(seq[:k]))
    current = recent[-1] if at_end else recent[0]
    if at_end:
        p0, p1 = recent[-2], recent[-1]
    else:
        p0, p1 = seq[1], seq[0]
    dy, dx = p1['center_y'] - p0['center_y'], p1['center_x'] - p0['center_x']
    if np.hypot(dy, dx) < 1e-6:
        angle = tangent_angle_from_item(current) + (0 if at_end else np.pi)
    else:
        angle = float(np.arctan2(dy, dx))
    lefts = [float(v.get('left_width_px', 0.5 * v['width_px'])) for v in recent]
    rights = [float(v.get('right_width_px', 0.5 * v['width_px'])) for v in recent]
    lt = float(np.median(np.diff(lefts))) if len(lefts) > 1 else 0.0
    rt = float(np.median(np.diff(rights))) if len(rights) > 1 else 0.0
    return dict(
        y=float(current['center_y']), x=float(current['center_x']), angle=angle,
        left_width=float(lefts[-1] if at_end else lefts[0]),
        right_width=float(rights[-1] if at_end else rights[0]),
        left_trend=lt, right_trend=rt,
        confidence=float(np.median([v['confidence'] for v in recent])),
        source=source, support_points=len(seq),
        source_grade=current.get('grade', 'C'),
    )


def _sequence_span(seq):
    if len(seq) < 2: return 0.0
    p = np.array([[v['center_y'], v['center_x']] for v in seq], float)
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))


def build_tracking_anchors(confirmed, ridge_only, edge_only):
    anchors = []
    # Confirmed anchors: require a short segment, never a single pixel.
    groups = {}
    for d in confirmed:
        key = (d.get('ridge_contour_id', -1), d.get('ridge_run_id', -1))
        groups.setdefault(key, []).append(d)
    for seq in groups.values():
        seq.sort(key=lambda v: v.get('ridge_arc_s', v.get('arc_s', 0.0)))
        if len(seq) < BLOCK_MIN_ANCHOR_POINTS or _sequence_span(seq) < BLOCK_MIN_ANCHOR_SPAN_PX:
            continue
        anchors += [_anchor_from_sequence(seq, True, 'confirmed'),
                    _anchor_from_sequence(seq, False, 'confirmed')]

    # Strong single-detector anchors are allowed only as multi-point, wide, confident runs.
    for samples, source in ((ridge_only, 'ridge_candidate'), (edge_only, 'edge_candidate')):
        groups = {}
        for d in samples:
            if d['confidence'] < BLOCK_STRONG_CANDIDATE_CONFIDENCE or d['width_px'] < BLOCK_TRIGGER_WIDTH_PX:
                continue
            key = (d.get('contour_id', -1), d.get('run_id', -1))
            groups.setdefault(key, []).append(d)
        for seq in groups.values():
            seq.sort(key=lambda v: v.get('arc_s', 0.0))
            if len(seq) < BLOCK_MIN_ANCHOR_POINTS + 1 or _sequence_span(seq) < 1.5 * BLOCK_MIN_ANCHOR_SPAN_PX:
                continue
            anchors += [_anchor_from_sequence(seq, True, source),
                        _anchor_from_sequence(seq, False, source)]

    anchors.sort(key=lambda a: (a['source'] == 'confirmed', a['confidence'], a['left_width'] + a['right_width']), reverse=True)
    kept = []
    for anchor in anchors:
        duplicate = False
        for prior in kept:
            dist = np.hypot(anchor['y'] - prior['y'], anchor['x'] - prior['x'])
            directed = abs((np.degrees(anchor['angle'] - prior['angle']) + 180.0) % 360.0 - 180.0)
            if dist < BLOCK_ANCHOR_NMS_RADIUS_PX and directed < 20.0:
                duplicate = True; break
        if not duplicate:
            kept.append(anchor)
        if len(kept) >= BLOCK_MAX_ANCHORS:
            break
    return kept


def _directed_orientation_score(theta_deg, coherency, angle):
    ty, tx = np.sin(angle), np.cos(angle)
    mismatch = axial_tangent_mismatch(theta_deg, ty, tx)
    agreement = float(np.exp(-((mismatch / 32.0) ** 2)))
    return float((0.35 + 0.65 * np.clip(coherency, 0, 1)) * agreement), mismatch


def _boundary_refine(grad, dense_dist, y, x, ny, nx):
    ts = np.linspace(-1.6, 1.6, 9)
    yy, xx = y + ny * ts, x + nx * ts
    g = _sample_line(grad, y, x, ny, nx, ts)
    d = _sample_line(dense_dist, y, x, ny, nx, ts)
    value = g * np.exp(-0.5 * (d / 1.8) ** 2)
    i = int(np.argmax(value))
    return float(yy[i]), float(xx[i])


def _evaluate_block_candidate(img, ridge_response, theta, coh, gy, gx, grad,
                              dense_dist, continuous_dist, junction_dist,
                              state, angle, left_width, right_width,
                              left_trend, right_trend, left_delta, right_delta,
                              grad_ref):
    ty, tx = np.sin(angle), np.cos(angle)
    y = state['y'] + BLOCK_STEP_PX * ty
    x = state['x'] + BLOCK_STEP_PX * tx
    if not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
        return None
    if any(np.hypot(y - p['center_y'], x - p['center_x']) < BLOCK_LOOP_RADIUS_PX
           for p in state['path'][:-4]):
        return None
    ny, nx = tx, -ty
    ly, lx = y - ny * left_width, x - nx * left_width
    ry, rx = y + ny * right_width, x + nx * right_width
    if not (1 <= ly < img.shape[0] - 1 and 1 <= lx < img.shape[1] - 1 and
            1 <= ry < img.shape[0] - 1 and 1 <= rx < img.shape[1] - 1):
        return None

    theta0, coh0 = sample_map(theta, y, x), sample_map(coh, y, x)
    orientation_score, mismatch = _directed_orientation_score(theta0, coh0, angle)
    ridge_score = sample_map(ridge_response, y, x)
    center_brightness = sample_map(img, y, x)

    ld = min(sample_map(dense_dist, ly, lx), sample_map(continuous_dist, ly, lx))
    rd = min(sample_map(dense_dist, ry, rx), sample_map(continuous_dist, ry, rx))
    left_support = float(np.exp(-0.5 * (ld / 2.0) ** 2))
    right_support = float(np.exp(-0.5 * (rd / 2.0) ** 2))
    edge_support = 0.5 * (left_support + right_support)

    # Boundary gradients in outward directions. Both are usually negative for a bright ribbon.
    lgy, lgx = sample_map(gy, ly, lx), sample_map(gx, ly, lx)
    rgy, rgx = sample_map(gy, ry, rx), sample_map(gx, ry, rx)
    dl = lgy * (-ny) + lgx * (-nx)
    dr = rgy * ny + rgx * nx
    strength = float(np.clip(0.5 * (abs(dl) + abs(dr)) / max(grad_ref, 1e-9), 0, 1))
    polarity = 1.0 if dl < 0 and dr < 0 else (0.55 if dl * dr > 0 else 0.30)
    gradient_score = 0.72 * strength + 0.28 * polarity

    ltheta = sample_map(theta, ly, lx); rtheta = sample_map(theta, ry, rx)
    boundary_orientation = 0.5 * (
        np.exp(-((axial_tangent_mismatch(ltheta, ty, tx) / 34.0) ** 2)) +
        np.exp(-((axial_tangent_mismatch(rtheta, ty, tx) / 34.0) ** 2)))

    outside_l = sample_map(img, ly - 2.0 * ny, lx - 2.0 * nx)
    outside_r = sample_map(img, ry + 2.0 * ny, rx + 2.0 * nx)
    contrast = center_brightness - 0.5 * (outside_l + outside_r)
    contrast_score = float(np.clip(contrast / 0.12, 0, 1))

    ts = np.linspace(-left_width + 1.0, right_width - 1.0,
                     max(5, int(round((left_width + right_width) / 3.0))))
    theta_inside = _sample_line(theta, y, x, ny, nx, ts)
    coh_inside = _sample_line(coh, y, x, ny, nx, ts)
    internal_orientation = orientation_profile_score(
        theta_inside, coh_inside, theta0, min_coherency=0.07, tolerance_deg=38.0)

    junction_nearness = sample_map(junction_dist, y, x)
    junction_score = float(np.clip(junction_nearness / BLOCK_JUNCTION_RADIUS_PX, 0, 1))
    turn_deg = abs(np.degrees((angle - state['angle'] + np.pi) % (2 * np.pi) - np.pi))
    width_trend_penalty = (abs(left_delta - state['left_trend']) +
                           abs(right_delta - state['right_trend'])) / max(BLOCK_WIDTH_DELTA_PX * 4.0, 1e-6)
    width_change_penalty = (abs(left_delta) + abs(right_delta)) / max(BLOCK_WIDTH_DELTA_PX * 4.0, 1e-6)
    asymmetry_change = (
        abs((left_width - right_width) - (state['left_width'] - state['right_width'])) /
        max(left_width + right_width, 1e-6)
    )

    local = (0.20 * ridge_score + 0.18 * orientation_score + 0.22 * edge_support +
             0.12 * gradient_score + 0.10 * contrast_score +
             0.08 * internal_orientation + 0.06 * boundary_orientation +
             0.04 * junction_score)
    local -= (0.055 * turn_deg / 45.0 + 0.055 * width_trend_penalty +
              0.035 * width_change_penalty + 0.04 * asymmetry_change)
    local = float(np.clip(local, 0, 1))

    ly, lx = _boundary_refine(grad, dense_dist, ly, lx, ny, nx)
    ry, rx = _boundary_refine(grad, dense_dist, ry, rx, ny, nx)
    weak = local < BLOCK_WEAK_LOCAL_SCORE
    item = dict(
        center_y=float(y), center_x=float(x), ym=float(y), xm=float(x),
        y1=ly, x1=lx, y2=ry, x2=rx,
        left_width_px=float(np.hypot(y - ly, x - lx)),
        right_width_px=float(np.hypot(y - ry, x - rx)),
        width_px=float(np.hypot(ry - ly, rx - lx)),
        tangent_angle_rad=float(angle), tangent_angle_deg=float(np.degrees(angle) % 180),
        curvature_deg_per_px=float(turn_deg / BLOCK_STEP_PX),
        ridge_response=float(ridge_score), orientation_score=float(orientation_score),
        orientation_mismatch_deg=float(mismatch), edge_support=float(edge_support),
        gradient_score=float(gradient_score), contrast_score=float(contrast_score),
        internal_orientation_score=float(internal_orientation),
        junction_score=float(junction_score), local_score=local,
        confidence=local, score=local, method='block', detector='block',
    )
    return dict(
        y=float(y), x=float(x), angle=float(angle),
        left_width=float(item['left_width_px']), right_width=float(item['right_width_px']),
        left_trend=float(left_trend), right_trend=float(right_trend),
        cumulative=float(state['cumulative'] + local), steps=int(state['steps'] + 1),
        weak_count=int(state['weak_count'] + 1 if weak else 0),
        path=state['path'] + [item],
        avg_edge=float((state['avg_edge'] * state['steps'] + edge_support) / (state['steps'] + 1)),
        avg_orientation=float((state['avg_orientation'] * state['steps'] + orientation_score) / (state['steps'] + 1)),
    )


def _state_objective(state):
    avg = state['cumulative'] / max(state['steps'], 1)
    length_reward = 0.08 * np.clip(state['steps'] / max(BLOCK_MIN_TRACK_STEPS * 2.0, 1), 0, 1)
    return float(avg + length_reward - 0.035 * state['weak_count'])


def run_single_block_track(img, ridge_response, theta, coh, gy, gx, grad,
                           dense_dist, continuous_dist, junction_dist,
                           anchor, max_width, track_id):
    min_half = BLOCK_MIN_HALF_WIDTH_PX
    max_half = max_width * 0.90
    grad_ref = float(np.percentile(grad[np.isfinite(grad)], 94))
    initial_item = dict(
        center_y=anchor['y'], center_x=anchor['x'], ym=anchor['y'], xm=anchor['x'],
        y1=anchor['y'], x1=anchor['x'], y2=anchor['y'], x2=anchor['x'],
        left_width_px=anchor['left_width'], right_width_px=anchor['right_width'],
        width_px=anchor['left_width'] + anchor['right_width'],
        tangent_angle_rad=anchor['angle'], tangent_angle_deg=float(np.degrees(anchor['angle']) % 180),
        confidence=anchor['confidence'], score=anchor['confidence'], local_score=anchor['confidence'],
        method='block', detector='block', anchor=True,
    )
    beam = [dict(
        y=anchor['y'], x=anchor['x'], angle=anchor['angle'],
        left_width=float(np.clip(anchor['left_width'], min_half, max_half)),
        right_width=float(np.clip(anchor['right_width'], min_half, max_half)),
        left_trend=float(anchor['left_trend']), right_trend=float(anchor['right_trend']),
        cumulative=float(anchor['confidence']), steps=1, weak_count=0,
        path=[initial_item], avg_edge=0.5, avg_orientation=0.5,
    )]
    terminals = []
    for _ in range(BLOCK_MAX_STEPS):
        expanded = []
        for state in beam:
            # Evaluate all requested turn blocks cheaply first, then expand only the best angles.
            angle_candidates = []
            for turn in BLOCK_TURN_DEGREES:
                angle = state['angle'] + np.deg2rad(turn)
                y = state['y'] + BLOCK_STEP_PX * np.sin(angle)
                x = state['x'] + BLOCK_STEP_PX * np.cos(angle)
                if not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
                    continue
                ori, _ = _directed_orientation_score(sample_map(theta, y, x), sample_map(coh, y, x), angle)
                preliminary = 0.55 * sample_map(ridge_response, y, x) + 0.45 * ori - 0.05 * abs(turn) / 45.0
                angle_candidates.append((preliminary, angle))
            angle_candidates.sort(reverse=True, key=lambda v: v[0])
            for _, angle in angle_candidates[:BLOCK_ANGLE_BRANCHES]:
                left_options = propose_half_widths(state['left_width'], state['left_trend'], min_half, max_half)
                right_options = propose_half_widths(state['right_width'], state['right_trend'], min_half, max_half)
                # Seven representative width combinations instead of the full Cartesian product.
                combos = representative_width_combos(left_options, right_options)
                seen = set()
                for L, R in combos:
                    key = (round(L[0], 2), round(R[0], 2))
                    if key in seen: continue
                    seen.add(key)
                    candidate = _evaluate_block_candidate(
                        img, ridge_response, theta, coh, gy, gx, grad,
                        dense_dist, continuous_dist, junction_dist,
                        state, angle, L[0], R[0], L[1], R[1], L[2], R[2], grad_ref)
                    if candidate is not None:
                        if candidate['weak_count'] > BLOCK_MAX_WEAK_STEPS:
                            terminals.append(candidate)
                        else:
                            expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(key=_state_objective, reverse=True)
        dedup, bins = [], set()
        for state in expanded:
            key = (int(round(state['y'] / 2.0)), int(round(state['x'] / 2.0)),
                   int(round((np.degrees(state['angle']) % 180) / 10.0)),
                   int(round(state['left_width'] / 2.0)), int(round(state['right_width'] / 2.0)))
            if key in bins: continue
            bins.add(key); dedup.append(state)
            if len(dedup) >= BLOCK_BEAM_WIDTH: break
        beam = dedup
        terminals.extend(beam)
    if not terminals:
        return [], dict(track_id=track_id, grade='C', reason='no_track')
    valid = [s for s in terminals if s['steps'] >= 2]
    best = max(valid, key=_state_objective)
    avg_conf = best['cumulative'] / max(best['steps'], 1)
    length = (best['steps'] - 1) * BLOCK_STEP_PX
    grade = ('B' if best['steps'] >= BLOCK_MIN_TRACK_STEPS and
             length >= BLOCK_MIN_TRACK_LENGTH_PX and
             avg_conf >= BLOCK_GRADE_B_MIN_AVG_CONFIDENCE and
             best['avg_edge'] >= BLOCK_GRADE_B_MIN_EDGE_SUPPORT and
             best['avg_orientation'] >= BLOCK_GRADE_B_MIN_ORIENTATION else 'C')
    path = []
    for i, item in enumerate(best['path'][1::BLOCK_PATH_SAMPLE_STRIDE]):
        d = canonicalize_measurement(item, 'block')
        d['grade'] = grade; d['track_id'] = track_id; d['track_step'] = i
        d['anchor_source'] = anchor['source']; d['track_length_px'] = float(length)
        d['track_avg_confidence'] = float(avg_conf)
        d['track_avg_edge_support'] = float(best['avg_edge'])
        d['track_avg_orientation'] = float(best['avg_orientation'])
        d['confidence'] = float(np.clip(0.68 * d['local_score'] + 0.32 * avg_conf, 0, 1))
        path.append(d)
    meta = dict(track_id=track_id, grade=grade, anchor_source=anchor['source'],
                n_steps=best['steps'], track_length_px=float(length),
                avg_confidence=float(avg_conf), avg_edge_support=float(best['avg_edge']),
                avg_orientation=float(best['avg_orientation']))
    return path, meta


def run_adaptive_block_tracks(img, ridge_response, theta, coh, gy, gx, grad,
                              continuous_edge, dense_edge, junction_labels,
                              anchors, max_width):
    dense_dist = ndimage.distance_transform_edt(~dense_edge)
    continuous_dist = ndimage.distance_transform_edt(~continuous_edge)
    junction_dist = ndimage.distance_transform_edt(~(junction_labels > 0)) if junction_labels.max() > 0 else np.full(img.shape, 999.0)
    grade_b, grade_c, metadata = [], [], []
    for track_id, anchor in enumerate(anchors, 1):
        path, meta = run_single_block_track(
            img, ridge_response, theta, coh, gy, gx, grad,
            dense_dist, continuous_dist, junction_dist,
            anchor, max_width, track_id)
        metadata.append(meta)
        (grade_b if meta['grade'] == 'B' else grade_c).extend(path)
    return grade_b, grade_c, metadata


def merge_final_measurements(grade_a, grade_b, grade_c):
    # Only A/B enter final statistics. Grade always outranks confidence for duplicates.
    accepted = [canonicalize_measurement(v) for v in list(grade_a) + list(grade_b)]
    accepted.sort(key=lambda d: (GRADE_PRIORITY.get(d.get('grade', 'C'), 0), d['confidence']), reverse=True)
    merged = []
    for cand in accepted:
        duplicate = next((i for i, old in enumerate(merged) if are_duplicate_measurements(cand, old)), None)
        if duplicate is None:
            merged.append(cand)
        else:
            old = merged[duplicate]
            if GRADE_PRIORITY.get(cand.get('grade', 'C'), 0) > GRADE_PRIORITY.get(old.get('grade', 'C'), 0):
                merged[duplicate] = cand
            elif cand.get('grade') == old.get('grade') and cand['confidence'] > old['confidence']:
                merged[duplicate] = cand
    candidates = []
    for cand in sorted([canonicalize_measurement(v) for v in grade_c], key=lambda d: d['confidence'], reverse=True):
        if any(are_duplicate_measurements(cand, old) for old in merged):
            continue
        if any(are_duplicate_measurements(cand, old) for old in candidates):
            continue
        candidates.append(cand)
    return merged, candidates


def _draw_center_paths(ax, samples, color, label, lw=1.0):
    if not samples: return
    groups = {}
    for d in samples:
        key = (d.get('track_id', d.get('contour_id', -1)), d.get('run_id', -1))
        groups.setdefault(key, []).append(d)
    first = True
    for seq in groups.values():
        seq.sort(key=lambda d: d.get('track_step', d.get('arc_s', 0)))
        xy = np.array([[d['center_x'], d['center_y']] for d in seq])
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, alpha=0.9,
                    label=label if first else None)
            first = False


def draw_consensus_block_result(img, ridge_response, ridge_centerline,
                                edge_samples, ridge_samples, confirmed,
                                block_b, block_c, final_samples, candidates,
                                name, output_png):
    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    for ax in axes.ravel():
        ax.imshow(img, cmap='gray'); ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis('off')

    axes[0, 0].imshow(ridge_response, cmap='magma', alpha=0.48)
    axes[0, 0].imshow(np.ma.masked_where(~ridge_centerline, ridge_centerline), cmap='winter', alpha=0.95)
    axes[0, 0].set_title(f'1. bright multi-scale ridges\ncenterline pixels={int(ridge_centerline.sum())}')

    add_chords(axes[0, 1], edge_samples, color='cyan', linewidth=0.9, label='edge detector')
    axes[0, 1].set_title(f'2. edge-pair detector\nn={len(edge_samples)}')

    add_chords(axes[0, 2], ridge_samples, color='magenta', linewidth=0.9, label='ridge detector')
    axes[0, 2].set_title(f'3. ridge + bilateral-edge detector\nn={len(ridge_samples)}')

    add_chords(axes[0, 3], confirmed, color='yellow', linewidth=1.5, label='Grade A')
    axes[0, 3].set_title(f'4. detector consensus anchors\nGrade A n={len(confirmed)}')

    _draw_center_paths(axes[1, 0], block_b, 'lime', 'Grade B block track', 1.4)
    _draw_center_paths(axes[1, 0], block_c, 'orange', 'Grade C block track', 0.8)
    add_chords(axes[1, 0], block_b[::max(1, len(block_b)//250 or 1)], color='lime', linewidth=0.7)
    axes[1, 0].legend(fontsize=7, loc='upper right')
    axes[1, 0].set_title(f'5. adaptive ribbon-block tracks\nB={len(block_b)}, C={len(block_c)}')

    grade_a = [d for d in final_samples if d.get('grade') == 'A']
    grade_b = [d for d in final_samples if d.get('grade') == 'B']
    add_chords(axes[1, 1], grade_a, color='yellow', linewidth=1.4, label='A: both detectors')
    add_chords(axes[1, 1], grade_b, color='lime', linewidth=1.1, label='B: detector + block')
    axes[1, 1].legend(fontsize=7, loc='upper right')
    median = np.median([d['width_px'] for d in final_samples]) if final_samples else np.nan
    axes[1, 1].set_title(f'6. accepted local-normal thickness\nn={len(final_samples)}, median={median:.2f}px')

    show_c = sorted(candidates, key=lambda d: d['confidence'], reverse=True)[:1000]
    add_chords(axes[1, 2], show_c, color='orange', linewidth=0.6, label='Grade C candidate')
    axes[1, 2].set_title(f'7. candidates only (excluded from statistics)\nshown={len(show_c)}/{len(candidates)}')

    axes[1, 3].clear(); axes[1, 3].axis('on'); axes[1, 3].set_aspect('auto')
    for grade, color in (('A', 'gold'), ('B', 'tab:green')):
        vals = [d['width_px'] for d in final_samples if d.get('grade') == grade]
        if vals:
            axes[1, 3].hist(vals, bins='auto', alpha=0.58, color=color, label=f'{grade} (n={len(vals)})')
    axes[1, 3].set_xlabel('thickness (px)'); axes[1, 3].set_ylabel('accepted measurements')
    axes[1, 3].legend(fontsize=8); axes[1, 3].set_title('8. final thickness distribution')

    fig.suptitle(f'{name} | ridge–edge consensus + adaptive ribbon blocks', fontsize=13)
    fig.tight_layout(); fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches='tight')
    if SHOW_INLINE: plt.show()
    plt.close(fig)


def process_one(path):
    img, cut = prepare_image(path)
    gy, gx, grad, theta, coh, energy = orientation_fields(img)
    continuous_edge = build_continuous_edges(img, coh, energy)
    dense_edge, dense_threshold = build_dense_edge_candidates(grad, gy, gx, coh, energy)
    edge_curves, edge_junctions = trace_edge_curves(continuous_edge)

    # Existing edge-based detector remains unchanged and supplies one independent vote.
    thin_samples, thin_rejected, thin_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, edge_curves, edge_junctions, 'thin')
    thick_samples, thick_rejected, thick_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, edge_curves, edge_junctions, 'thick')
    curved_samples, curved_rejected, curved_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge, edge_curves, edge_junctions, 'curved')
    edge_samples = [canonicalize_measurement(v, 'edge') for v in
                    merge_method_results(thin_samples, thick_samples, curved_samples)]

    ridge_response, ridge_centerline, ridge_curves, ridge_junctions = build_bright_ridge_centerlines(img)
    ridge_samples, ridge_rejected, ridge_max = measure_ridge_detector(
        img, ridge_response, ridge_curves, gy, gx, grad, theta, coh, continuous_edge, dense_edge)

    confirmed, ridge_only, edge_only = match_detector_consensus(ridge_samples, edge_samples)
    anchors = build_tracking_anchors(confirmed, ridge_only, edge_only)
    block_b, block_c, track_meta = run_adaptive_block_tracks(
        img, ridge_response, theta, coh, gy, gx, grad,
        continuous_edge, dense_edge, edge_junctions, anchors,
        max(thin_max, thick_max, curved_max, ridge_max))

    final_samples, candidates = merge_final_measurements(
        confirmed, block_b, ridge_only + edge_only + block_c)

    # Data tables.
    final_df = pd.DataFrame(final_samples)
    if len(final_df):
        final_df.insert(0, 'file', path.name); final_df['status'] = 'accepted'
        nm_px = NM_PER_PX.get(path.name)
        final_df['nm_per_px'] = nm_px if nm_px is not None else np.nan
        final_df['thickness_nm'] = final_df['width_px'] * nm_px if nm_px is not None else np.nan
    candidate_df = pd.DataFrame(candidates)
    if len(candidate_df):
        candidate_df.insert(0, 'file', path.name); candidate_df['status'] = 'candidate_only'
    detector_df = pd.DataFrame(edge_samples + ridge_samples)
    if len(detector_df): detector_df.insert(0, 'file', path.name)
    rejection_df = pd.DataFrame(thin_rejected + thick_rejected + curved_rejected + ridge_rejected)
    if len(rejection_df): rejection_df.insert(0, 'file', path.name)
    track_df = pd.DataFrame(track_meta)
    if len(track_df): track_df.insert(0, 'file', path.name)

    output_png = OUTPUT_DIR / f'{path.stem}_consensus_block_thickness.png'
    draw_consensus_block_result(
        img, ridge_response, ridge_centerline, edge_samples, ridge_samples,
        confirmed, block_b, block_c, final_samples, candidates,
        path.name, output_png)

    final_df.to_csv(OUTPUT_DIR / f'{path.stem}_accepted_A_B_thickness.csv', index=False)
    candidate_df.to_csv(OUTPUT_DIR / f'{path.stem}_grade_C_candidates.csv', index=False)
    detector_df.to_csv(OUTPUT_DIR / f'{path.stem}_raw_detector_measurements.csv', index=False)
    track_df.to_csv(OUTPUT_DIR / f'{path.stem}_block_track_summary.csv', index=False)
    rejection_df.to_csv(OUTPUT_DIR / f'{path.stem}_rejected_runs.csv', index=False)

    widths = final_df['width_px'].to_numpy(float) if len(final_df) else np.array([])
    grades = final_df['grade'].value_counts().to_dict() if len(final_df) else {}
    summary = dict(
        file=path.name, cropped_height=cut, image_height=img.shape[0], image_width=img.shape[1],
        edge_detector_measurements=len(edge_samples), ridge_detector_measurements=len(ridge_samples),
        grade_A_measurements=int(grades.get('A', 0)), grade_B_measurements=int(grades.get('B', 0)),
        grade_C_candidates=len(candidate_df), block_anchors=len(anchors),
        grade_B_tracks=sum(m.get('grade') == 'B' for m in track_meta),
        grade_C_tracks=sum(m.get('grade') == 'C' for m in track_meta),
        accepted_measurements=len(final_df),
        median_thickness_px=float(np.median(widths)) if len(widths) else np.nan,
        p25_thickness_px=float(np.percentile(widths, 25)) if len(widths) else np.nan,
        p75_thickness_px=float(np.percentile(widths, 75)) if len(widths) else np.nan,
        max_thickness_px=float(np.max(widths)) if len(widths) else np.nan,
        dense_gradient_threshold=dense_threshold,
    )
    return summary, final_df, candidate_df


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f'Folder not found: {IMAGE_DIR}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files, missing = [], []
    for name in TARGET_FILES:
        p = IMAGE_DIR / name
        (files if p.is_file() else missing).append(p if p.is_file() else name)
    if missing:
        raise FileNotFoundError('Requested image(s) not found: ' + ', '.join(map(str, missing)))
    print(f'Input : {IMAGE_DIR}\nOutput: {OUTPUT_DIR}')
    print('Images: ' + ', '.join(p.name for p in files))
    summaries, accepted_frames, candidate_frames = [], [], []
    for i, path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] {path.name}')
        try:
            summary, accepted, candidates = process_one(path)
            summaries.append(summary)
            if len(accepted): accepted_frames.append(accepted)
            if len(candidates): candidate_frames.append(candidates)
            print(f"    A={summary['grade_A_measurements']}, B={summary['grade_B_measurements']}, "
                  f"C={summary['grade_C_candidates']}, accepted={summary['accepted_measurements']}, "
                  f"median={summary['median_thickness_px']:.2f}px")
        except Exception as exc:
            warnings.warn(f'{path.name}: {type(exc).__name__}: {exc}')
            summaries.append(dict(file=path.name, error=f'{type(exc).__name__}: {exc}'))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / 'consensus_block_summary.csv', index=False)
    if accepted_frames:
        pd.concat(accepted_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_accepted_A_B_thickness.csv', index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_grade_C_candidates.csv', index=False)
    print('\nDone. Results saved under:')
    print(OUTPUT_DIR)
    display(summary_df) if 'display' in globals() else print(summary_df)




# ============================================================================
# Bundle-guard override
# Prevents several nearby thin fibres from being merged into one thick fibre.
# This section intentionally reuses the detectors/helpers defined above and only
# overrides the ridge, ribbon-block, visualization, and per-image orchestration.
# ============================================================================
TARGET_FILES = ['2-10.jpg', '2-11.jpg', '2-19.jpg']
OUTPUT_DIR = IMAGE_DIR.parent / f'{IMAGE_DIR.name}_edge_thickness_consensus_bundle_guard'

# Separate small-scale ridges reveal individual thin fibres inside a wide candidate.
RIDGE_SMALL_SIGMAS = (0.8, 1.2, 1.8, 2.5, 3.2)
RIDGE_SMALL_RESPONSE_PERCENTILE = 84.0
RIDGE_SMALL_MIN_COMPONENT_PX = 12

# Bundle guard. A wide ribbon is suspicious when two or more independent tests
# indicate multiple thin fibres: small-scale centerlines, locked thin detections,
# or a persistent peak-valley-peak intensity profile.
BUNDLE_CHECK_MIN_WIDTH_PX = 11.0
BUNDLE_THIN_MAX_WIDTH_PX = 18.0
BUNDLE_THIN_MIN_CONFIDENCE = 0.46
BUNDLE_THIN_TANGENT_WINDOW_PX = 8.0
BUNDLE_THIN_MAX_TANGENT_DIFF_DEG = 28.0
BUNDLE_THIN_CLUSTER_GAP_PX = 3.2
BUNDLE_SMALL_RIDGE_HIT_PX = 1.35
BUNDLE_SMALL_RIDGE_CLUSTER_GAP_PX = 2.4
BUNDLE_PROFILE_STEP_PX = 0.5
BUNDLE_PROFILE_PEAK_PROMINENCE = 0.10
BUNDLE_PROFILE_MIN_PEAK_SEPARATION_PX = 3.0
BUNDLE_PROFILE_MIN_VALLEY_DEPTH = 0.18
BUNDLE_PAIR_SCORE_PENALTY = 0.20
BUNDLE_MODERATE_SCORE = 0.42
BUNDLE_RUN_STRONG_FRACTION = 0.25
BUNDLE_RUN_MODERATE_FRACTION = 0.60
BUNDLE_BLOCK_HARD_VOTES = 2
BUNDLE_BLOCK_SCORE_PENALTY = 0.26
BUNDLE_BLOCK_MAX_AVG_SCORE = 0.40
BUNDLE_BLOCK_MAX_HIT_FRACTION = 0.35


def _normalise_ridge_response(response):
    response = np.asarray(response, np.float32)
    finite = np.isfinite(response)
    if not finite.any() or float(np.max(response[finite])) <= 0:
        return np.zeros_like(response, np.float32)
    lo, hi = np.percentile(response[finite], [5, 99.7])
    return np.clip((response - lo) / max(float(hi - lo), 1e-9), 0, 1).astype(np.float32)


def _ridge_centerline_from_response(img, response_n, percentile, min_component):
    finite = np.isfinite(response_n)
    if not finite.any() or float(np.max(response_n[finite])) <= 0:
        return np.zeros_like(img, bool), [], np.zeros_like(img, np.int32)
    rthr = float(np.percentile(response_n[finite], percentile))
    ithr = float(np.percentile(img[np.isfinite(img)], RIDGE_MIN_INTENSITY_PERCENTILE))
    mask = finite & (response_n >= rthr) & (img >= ithr)
    if RIDGE_CLOSE_RADIUS_PX > 0:
        mask = closing(mask, disk(RIDGE_CLOSE_RADIUS_PX))
    mask = _remove_small_objects_compat(mask.astype(bool), min_component)
    centerline = skeletonize(mask)
    centerline = remove_small_components(centerline, min_component)
    curves, junction_labels = trace_edge_curves(centerline)
    return centerline.astype(bool), curves, junction_labels


def build_bright_ridge_centerlines(img):
    """Return broad ridge hypotheses and a separate small-scale thin-fibre map."""
    broad = _normalise_ridge_response(
        sato(img, sigmas=RIDGE_SIGMAS, black_ridges=False, mode='reflect'))
    small = _normalise_ridge_response(
        sato(img, sigmas=RIDGE_SMALL_SIGMAS, black_ridges=False, mode='reflect'))
    broad_center, broad_curves, broad_junctions = _ridge_centerline_from_response(
        img, broad, RIDGE_RESPONSE_PERCENTILE, RIDGE_MIN_COMPONENT_PX)
    small_center, small_curves, small_junctions = _ridge_centerline_from_response(
        img, small, RIDGE_SMALL_RESPONSE_PERCENTILE, RIDGE_SMALL_MIN_COMPONENT_PX)
    return (broad, broad_center, broad_curves, broad_junctions,
            small, small_center, small_curves, small_junctions)


def count_1d_clusters(values, gap_px):
    values = np.sort(np.asarray(values, float))
    if values.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(values) > float(gap_px)))


def profile_bundle_features(profile, step_px=0.5):
    """Detect a repeatable peak-valley-peak signature of multiple thin fibres."""
    p = np.asarray(profile, np.float32)
    p = p[np.isfinite(p)]
    if p.size < 7:
        return dict(peak_count=0, max_valley_depth=0.0, profile_bundle=False)
    sigma_samples = max(0.8, 0.8 / max(float(step_px), 1e-6))
    sm = ndimage.gaussian_filter1d(p, sigma_samples, mode='nearest')
    lo, hi = np.percentile(sm, [5, 95])
    if hi - lo < 1e-6:
        return dict(peak_count=0, max_valley_depth=0.0, profile_bundle=False)
    z = np.clip((sm - lo) / (hi - lo), 0, 1)
    min_distance = max(1, int(round(BUNDLE_PROFILE_MIN_PEAK_SEPARATION_PX / max(step_px, 1e-6))))
    peaks, _ = find_peaks(z, prominence=BUNDLE_PROFILE_PEAK_PROMINENCE, distance=min_distance)
    max_depth = 0.0
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b <= a + 1:
            continue
        valley = float(np.min(z[a:b + 1]))
        depth = float(min(z[a], z[b]) - valley)
        max_depth = max(max_depth, depth)
    is_bundle = bool(len(peaks) >= 2 and max_depth >= BUNDLE_PROFILE_MIN_VALLEY_DEPTH)
    return dict(peak_count=int(len(peaks)), max_valley_depth=float(max_depth),
                profile_bundle=is_bundle)


def build_thin_lock_model(edge_samples):
    samples = []
    for item in edge_samples:
        d = canonicalize_measurement(item, 'edge')
        if (d['width_px'] <= BUNDLE_THIN_MAX_WIDTH_PX and
                d['confidence'] >= BUNDLE_THIN_MIN_CONFIDENCE):
            samples.append(d)
    centers = np.array([[d['center_y'], d['center_x']] for d in samples], float)
    tree = cKDTree(centers) if len(centers) else None
    return dict(samples=samples, centers=centers, tree=tree)


def _thin_lock_cluster_count(model, y, x, ty, tx, left_width, right_width):
    tree = model.get('tree') if model else None
    samples = model.get('samples', []) if model else []
    if tree is None or not samples:
        return 0
    radius = max(left_width, right_width) + BUNDLE_THIN_TANGENT_WINDOW_PX + 2.0
    indices = tree.query_ball_point([y, x], radius)
    if not indices:
        return 0
    ny, nx = tx, -ty
    offsets = []
    candidate_angle = float(np.degrees(np.arctan2(ty, tx)) % 180.0)
    for idx in indices:
        d = samples[idx]
        dy, dx = d['center_y'] - y, d['center_x'] - x
        tangent_offset = dy * ty + dx * tx
        normal_offset = dy * ny + dx * nx
        if abs(tangent_offset) > BUNDLE_THIN_TANGENT_WINDOW_PX:
            continue
        if not (-left_width + 1.0 <= normal_offset <= right_width - 1.0):
            continue
        thin_angle = float(np.degrees(tangent_angle_from_item(d)) % 180.0)
        if axial_angle_diff_deg(candidate_angle, thin_angle) > BUNDLE_THIN_MAX_TANGENT_DIFF_DEG:
            continue
        offsets.append(normal_offset)
    return count_1d_clusters(offsets, BUNDLE_THIN_CLUSTER_GAP_PX)


def bundle_evidence_at_ribbon(img, small_ridge_dist, thin_lock_model,
                              y, x, ty, tx, left_width, right_width):
    width = float(left_width + right_width)
    empty = dict(bundle_score=0.0, bundle_votes=0, small_ridge_count=0,
                 thin_lock_count=0, profile_peak_count=0,
                 profile_valley_depth=0.0, profile_bundle=False)
    if width < BUNDLE_CHECK_MIN_WIDTH_PX:
        return empty
    ny, nx = tx, -ty
    ts = np.arange(-left_width + 0.5, right_width - 0.5 + 1e-6,
                   BUNDLE_PROFILE_STEP_PX, dtype=np.float32)
    if ts.size < 7:
        return empty
    profile = _sample_line(img, y, x, ny, nx, ts)
    pf = profile_bundle_features(profile, BUNDLE_PROFILE_STEP_PX)

    small_dist_profile = _sample_line(small_ridge_dist, y, x, ny, nx, ts)
    small_hits = ts[small_dist_profile <= BUNDLE_SMALL_RIDGE_HIT_PX]
    small_count = count_1d_clusters(small_hits, BUNDLE_SMALL_RIDGE_CLUSTER_GAP_PX)
    thin_count = _thin_lock_cluster_count(
        thin_lock_model, y, x, ty, tx, left_width, right_width)

    small_vote = small_count >= 2
    thin_vote = thin_count >= 2
    profile_vote = bool(pf['profile_bundle'])
    votes = int(small_vote) + int(thin_vote) + int(profile_vote)
    small_strength = float(np.clip((small_count - 1) / 2.0, 0, 1))
    thin_strength = float(np.clip((thin_count - 1) / 2.0, 0, 1))
    profile_strength = float(np.clip(pf['max_valley_depth'] /
                                     max(2.0 * BUNDLE_PROFILE_MIN_VALLEY_DEPTH, 1e-6), 0, 1))
    score = float(0.38 * small_strength + 0.36 * thin_strength + 0.26 * profile_strength)
    return dict(bundle_score=score, bundle_votes=votes,
                small_ridge_count=int(small_count), thin_lock_count=int(thin_count),
                profile_peak_count=int(pf['peak_count']),
                profile_valley_depth=float(pf['max_valley_depth']),
                profile_bundle=profile_vote)


def ridge_pairs_at_sample(img, ridge_response, gy, gx, grad, theta, coh, edge_dist,
                          small_ridge_dist, thin_lock_model,
                          y, x, ty, tx, min_width, max_width, grad_ref):
    ny, nx = tx, -ty
    max_half = min(max_width, max_width * 0.80)
    left = _ridge_side_candidates(img, gy, gx, grad, theta, coh, edge_dist,
                                  y, x, ty, tx, -ny, -nx, max_half, grad_ref)
    right = _ridge_side_candidates(img, gy, gx, grad, theta, coh, edge_dist,
                                   y, x, ty, tx, ny, nx, max_half, grad_ref)
    if not left or not right:
        return []
    center_theta = sample_map(theta, y, x)
    center_coh = sample_map(coh, y, x)
    center_tangent = float(np.clip(1.0 - axial_tangent_mismatch(center_theta, ty, tx) / 45.0, 0, 1))
    ridge_score = sample_map(ridge_response, y, x)
    candidates = []
    for L in left:
        for R in right:
            width = L['half_width_px'] + R['half_width_px']
            if not (min_width <= width <= max_width):
                continue
            ts = np.linspace(-L['half_width_px'] + 1.0,
                             R['half_width_px'] - 1.0,
                             max(5, int(round(width / 2.0))))
            theta_inside = _sample_line(theta, y, x, ny, nx, ts)
            coh_inside = _sample_line(coh, y, x, ny, nx, ts)
            orientation_score = orientation_profile_score(
                theta_inside, coh_inside, center_theta,
                min_coherency=0.08, tolerance_deg=34.0)
            asymmetry = abs(L['half_width_px'] - R['half_width_px']) / max(width, 1e-9)
            asymmetry_score = float(np.exp(-((asymmetry / 0.42) ** 2)))
            boundary_score = 0.5 * (L['score'] + R['score'])
            raw_pair_score = float(
                0.35 * boundary_score + 0.20 * ridge_score +
                0.16 * orientation_score + 0.12 * center_tangent +
                0.10 * np.clip(center_coh, 0, 1) + 0.07 * asymmetry_score)
            if raw_pair_score < RIDGE_MIN_PAIR_SCORE:
                continue
            bundle = bundle_evidence_at_ribbon(
                img, small_ridge_dist, thin_lock_model, y, x, ty, tx,
                L['half_width_px'], R['half_width_px'])
            penalty = BUNDLE_PAIR_SCORE_PENALTY * bundle['bundle_score']
            if bundle['bundle_votes'] >= 2:
                penalty += 0.10
            pair_score = float(np.clip(raw_pair_score - penalty, 0, 1))
            candidates.append(dict(
                y1=L['y'], x1=L['x'], y2=R['y'], x2=R['x'],
                ym=float(y), xm=float(x), center_y=float(y), center_x=float(x),
                left_width_px=float(L['half_width_px']),
                right_width_px=float(R['half_width_px']),
                width_px=float(width), score=pair_score, confidence=pair_score,
                raw_pair_score=raw_pair_score,
                ridge_response=float(ridge_score),
                orientation_profile_score=float(orientation_score),
                center_tangent_score=float(center_tangent),
                left_edge_support=float(L['edge_support']),
                right_edge_support=float(R['edge_support']),
                edge_support=float(0.5 * (L['edge_support'] + R['edge_support'])),
                method='ridge', detector='ridge', **bundle,
            ))
    candidates.sort(key=lambda d: d['score'], reverse=True)
    return candidates[:10]


def measure_ridge_detector(img, ridge_response, centerline_curves,
                           gy, gx, grad, theta, coh, continuous_edge, dense_edge,
                           small_ridge_dist, thin_lock_model):
    search_edge = continuous_edge | dense_edge
    edge_dist = ndimage.distance_transform_edt(~search_edge)
    max_width = min(80.0, 0.16 * min(img.shape)) if RIDGE_MAX_WIDTH_PX is None else float(RIDGE_MAX_WIDTH_PX)
    min_width = RIDGE_MIN_WIDTH_PX
    grad_values = grad[search_edge]
    grad_ref = float(np.percentile(grad_values, 80)) if grad_values.size else float(np.percentile(grad, 94))
    grad_ref = max(grad_ref, 1e-8)
    accepted, rejected, bundle_rejected = [], [], []
    run_id = 0
    for contour_id, raw in enumerate(centerline_curves):
        path, arc = resample_polyline(raw, 1.0)
        if len(path) < RIDGE_MIN_COMPONENT_PX:
            continue
        sy = ndimage.gaussian_filter1d(path[:, 0], 2.0, mode='nearest')
        sx = ndimage.gaussian_filter1d(path[:, 1], 2.0, mode='nearest')
        ty, tx, directed, turn_rate, good = local_turn_rate_and_angle(sy, sx, arc)
        records = []
        for idx in np.arange(2, len(path) - 2, max(1, int(round(RIDGE_SAMPLE_SPACING_PX)))):
            y, x = float(sy[idx]), float(sx[idx])
            if not good[idx] or not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
                candidates = []
            else:
                candidates = ridge_pairs_at_sample(
                    img, ridge_response, gy, gx, grad, theta, coh, edge_dist,
                    small_ridge_dist, thin_lock_model,
                    y, x, float(ty[idx]), float(tx[idx]), min_width, max_width, grad_ref)
            records.append(dict(index=int(idx), arc_s=float(arc[idx]), y=y, x=x,
                                ty=float(ty[idx]), tx=float(tx[idx]),
                                tangent_angle_rad=float(directed[idx]),
                                local_turn_rate_deg_per_px=float(turn_rate[idx]),
                                candidates=candidates))
        chosen = viterbi_select_ridge(records)
        runs, current, gap = [], [], 0
        for record, candidate in zip(records, chosen):
            if candidate is None:
                if current:
                    gap += 1
                    if gap <= RIDGE_ALLOWED_GAP_SAMPLES:
                        current.append(None)
                    else:
                        while current and current[-1] is None:
                            current.pop()
                        if current:
                            runs.append(current)
                        current, gap = [], 0
                continue
            gap = 0
            item = dict(candidate, contour_id=contour_id, arc_s=record['arc_s'],
                        tangent_angle_rad=record['tangent_angle_rad'],
                        tangent_angle_deg=float(np.degrees(record['tangent_angle_rad']) % 180),
                        local_turn_rate_deg_per_px=record['local_turn_rate_deg_per_px'])
            valid_current = [v for v in current if v is not None]
            if valid_current:
                prior = valid_current[-1]
                width_jump = abs(item['width_px'] - prior['width_px']) / max(prior['width_px'], 1e-9)
                center_jump = np.hypot(item['center_y'] - prior['center_y'], item['center_x'] - prior['center_x'])
                if width_jump > 0.52 or center_jump > 3.5 * RIDGE_SAMPLE_SPACING_PX:
                    runs.append(current)
                    current = []
            current.append(item)
        if current:
            runs.append(current)

        for run in runs:
            run_id += 1
            clean = [v for v in run if v is not None]
            if not clean:
                continue
            length = clean[-1]['arc_s'] - clean[0]['arc_s'] if len(clean) > 1 else 0.0
            widths = [v['width_px'] for v in clean]
            cv = robust_cv(widths)
            median_score = float(np.median([v['score'] for v in clean]))
            strong_fraction = float(np.mean([v.get('bundle_votes', 0) >= 2 for v in clean]))
            moderate_fraction = float(np.mean([
                v.get('bundle_score', 0.0) >= BUNDLE_MODERATE_SCORE or v.get('bundle_votes', 0) >= 1
                for v in clean]))
            median_bundle = float(np.median([v.get('bundle_score', 0.0) for v in clean]))
            reason = None
            if strong_fraction >= BUNDLE_RUN_STRONG_FRACTION:
                reason = 'multiple_thin_fibres_bundle_strong'
            elif moderate_fraction >= BUNDLE_RUN_MODERATE_FRACTION:
                reason = 'multiple_thin_fibres_bundle_persistent'
            elif len(clean) < RIDGE_MIN_RUN_SAMPLES:
                reason = 'too_few_ridge_samples'
            elif length < RIDGE_MIN_RUN_LENGTH_PX:
                reason = 'ridge_run_too_short'
            elif cv > RIDGE_MAX_RUN_ROBUST_CV:
                reason = 'ridge_width_not_stable'
            elif median_score < RIDGE_MIN_PAIR_SCORE:
                reason = 'ridge_score_too_low'
            if reason:
                rejected.append(dict(detector='ridge', contour_id=contour_id, run_id=run_id,
                                     reason=reason, n_samples=len(clean), run_length_px=length,
                                     robust_cv=cv, median_score=median_score,
                                     strong_bundle_fraction=strong_fraction,
                                     moderate_bundle_fraction=moderate_fraction,
                                     median_bundle_score=median_bundle))
                if reason.startswith('multiple_thin_fibres'):
                    for item in clean:
                        d = canonicalize_measurement(item, 'ridge')
                        d['run_id'] = run_id
                        d['reject_reason'] = reason
                        d['grade'] = 'C'
                        bundle_rejected.append(d)
                continue
            run_stability = np.clip(1.0 - cv / RIDGE_MAX_RUN_ROBUST_CV, 0, 1)
            length_score = np.clip(length / (2.0 * RIDGE_MIN_RUN_LENGTH_PX), 0, 1)
            for item in clean:
                item['run_id'] = run_id
                item['run_length_px'] = float(length)
                item['run_robust_cv'] = float(cv)
                bundle_cleanliness = 1.0 - item.get('bundle_score', 0.0)
                item['confidence'] = float(
                    0.64 * item['score'] + 0.16 * run_stability +
                    0.08 * length_score + 0.12 * bundle_cleanliness)
                item['grade'] = 'C'
                accepted.append(canonicalize_measurement(item, 'ridge'))
    return accepted, rejected, max_width, bundle_rejected


def _evaluate_block_candidate(img, ridge_response, theta, coh, gy, gx, grad,
                              dense_dist, continuous_dist, junction_dist,
                              small_ridge_dist, thin_lock_model,
                              state, angle, left_width, right_width,
                              left_trend, right_trend, left_delta, right_delta,
                              grad_ref):
    ty, tx = np.sin(angle), np.cos(angle)
    y = state['y'] + BLOCK_STEP_PX * ty
    x = state['x'] + BLOCK_STEP_PX * tx
    if not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
        return None
    if any(np.hypot(y - p['center_y'], x - p['center_x']) < BLOCK_LOOP_RADIUS_PX
           for p in state['path'][:-4]):
        return None
    ny, nx = tx, -ty
    ly, lx = y - ny * left_width, x - nx * left_width
    ry, rx = y + ny * right_width, x + nx * right_width
    if not (1 <= ly < img.shape[0] - 1 and 1 <= lx < img.shape[1] - 1 and
            1 <= ry < img.shape[0] - 1 and 1 <= rx < img.shape[1] - 1):
        return None

    bundle = bundle_evidence_at_ribbon(
        img, small_ridge_dist, thin_lock_model, y, x, ty, tx, left_width, right_width)
    if bundle['bundle_votes'] >= BUNDLE_BLOCK_HARD_VOTES:
        return None

    theta0, coh0 = sample_map(theta, y, x), sample_map(coh, y, x)
    orientation_score, mismatch = _directed_orientation_score(theta0, coh0, angle)
    ridge_score = sample_map(ridge_response, y, x)
    center_brightness = sample_map(img, y, x)
    ld = min(sample_map(dense_dist, ly, lx), sample_map(continuous_dist, ly, lx))
    rd = min(sample_map(dense_dist, ry, rx), sample_map(continuous_dist, ry, rx))
    left_support = float(np.exp(-0.5 * (ld / 2.0) ** 2))
    right_support = float(np.exp(-0.5 * (rd / 2.0) ** 2))
    edge_support = 0.5 * (left_support + right_support)

    lgy, lgx = sample_map(gy, ly, lx), sample_map(gx, ly, lx)
    rgy, rgx = sample_map(gy, ry, rx), sample_map(gx, ry, rx)
    dl = lgy * (-ny) + lgx * (-nx)
    dr = rgy * ny + rgx * nx
    strength = float(np.clip(0.5 * (abs(dl) + abs(dr)) / max(grad_ref, 1e-9), 0, 1))
    polarity = 1.0 if dl < 0 and dr < 0 else (0.55 if dl * dr > 0 else 0.30)
    gradient_score = 0.72 * strength + 0.28 * polarity
    ltheta = sample_map(theta, ly, lx)
    rtheta = sample_map(theta, ry, rx)
    boundary_orientation = 0.5 * (
        np.exp(-((axial_tangent_mismatch(ltheta, ty, tx) / 34.0) ** 2)) +
        np.exp(-((axial_tangent_mismatch(rtheta, ty, tx) / 34.0) ** 2)))
    outside_l = sample_map(img, ly - 2.0 * ny, lx - 2.0 * nx)
    outside_r = sample_map(img, ry + 2.0 * ny, rx + 2.0 * nx)
    contrast = center_brightness - 0.5 * (outside_l + outside_r)
    contrast_score = float(np.clip(contrast / 0.12, 0, 1))
    ts = np.linspace(-left_width + 1.0, right_width - 1.0,
                     max(5, int(round((left_width + right_width) / 3.0))))
    theta_inside = _sample_line(theta, y, x, ny, nx, ts)
    coh_inside = _sample_line(coh, y, x, ny, nx, ts)
    internal_orientation = orientation_profile_score(
        theta_inside, coh_inside, theta0, min_coherency=0.07, tolerance_deg=38.0)
    junction_nearness = sample_map(junction_dist, y, x)
    junction_score = float(np.clip(junction_nearness / BLOCK_JUNCTION_RADIUS_PX, 0, 1))
    turn_deg = abs(np.degrees((angle - state['angle'] + np.pi) % (2 * np.pi) - np.pi))
    width_trend_penalty = (abs(left_delta - state['left_trend']) +
                           abs(right_delta - state['right_trend'])) / max(BLOCK_WIDTH_DELTA_PX * 4.0, 1e-6)
    width_change_penalty = (abs(left_delta) + abs(right_delta)) / max(BLOCK_WIDTH_DELTA_PX * 4.0, 1e-6)
    asymmetry_change = abs((left_width - right_width) -
                           (state['left_width'] - state['right_width'])) / max(left_width + right_width, 1e-6)

    local = (0.20 * ridge_score + 0.18 * orientation_score + 0.22 * edge_support +
             0.12 * gradient_score + 0.10 * contrast_score +
             0.08 * internal_orientation + 0.06 * boundary_orientation +
             0.04 * junction_score)
    local -= (0.055 * turn_deg / 45.0 + 0.055 * width_trend_penalty +
              0.035 * width_change_penalty + 0.04 * asymmetry_change)
    expanding = max(0.0, left_delta) + max(0.0, right_delta)
    local -= BUNDLE_BLOCK_SCORE_PENALTY * bundle['bundle_score']
    if expanding > 0 and bundle['bundle_votes'] >= 1:
        local -= 0.08 * bundle['bundle_score']
    local = float(np.clip(local, 0, 1))

    ly, lx = _boundary_refine(grad, dense_dist, ly, lx, ny, nx)
    ry, rx = _boundary_refine(grad, dense_dist, ry, rx, ny, nx)
    weak = local < BLOCK_WEAK_LOCAL_SCORE
    item = dict(
        center_y=float(y), center_x=float(x), ym=float(y), xm=float(x),
        y1=ly, x1=lx, y2=ry, x2=rx,
        left_width_px=float(np.hypot(y - ly, x - lx)),
        right_width_px=float(np.hypot(y - ry, x - rx)),
        width_px=float(np.hypot(ry - ly, rx - lx)),
        tangent_angle_rad=float(angle), tangent_angle_deg=float(np.degrees(angle) % 180),
        curvature_deg_per_px=float(turn_deg / BLOCK_STEP_PX),
        ridge_response=float(ridge_score), orientation_score=float(orientation_score),
        orientation_mismatch_deg=float(mismatch), edge_support=float(edge_support),
        gradient_score=float(gradient_score), contrast_score=float(contrast_score),
        internal_orientation_score=float(internal_orientation),
        junction_score=float(junction_score), local_score=local,
        confidence=local, score=local, method='block', detector='block', **bundle,
    )
    prior_steps = state['steps']
    return dict(
        y=float(y), x=float(x), angle=float(angle),
        left_width=float(item['left_width_px']), right_width=float(item['right_width_px']),
        left_trend=float(left_trend), right_trend=float(right_trend),
        cumulative=float(state['cumulative'] + local), steps=int(prior_steps + 1),
        weak_count=int(state['weak_count'] + 1 if weak else 0),
        path=state['path'] + [item],
        avg_edge=float((state['avg_edge'] * prior_steps + edge_support) / (prior_steps + 1)),
        avg_orientation=float((state['avg_orientation'] * prior_steps + orientation_score) / (prior_steps + 1)),
        avg_bundle=float((state.get('avg_bundle', 0.0) * prior_steps + bundle['bundle_score']) / (prior_steps + 1)),
        bundle_hits=int(state.get('bundle_hits', 0) + (bundle['bundle_votes'] >= 1)),
    )


def _state_objective(state):
    avg = state['cumulative'] / max(state['steps'], 1)
    length_reward = 0.08 * np.clip(state['steps'] / max(BLOCK_MIN_TRACK_STEPS * 2.0, 1), 0, 1)
    return float(avg + length_reward - 0.035 * state['weak_count'] -
                 0.16 * state.get('avg_bundle', 0.0))


def run_single_block_track(img, ridge_response, theta, coh, gy, gx, grad,
                           dense_dist, continuous_dist, junction_dist,
                           small_ridge_dist, thin_lock_model,
                           anchor, max_width, track_id):
    min_half = BLOCK_MIN_HALF_WIDTH_PX
    max_half = max_width * 0.90
    grad_ref = float(np.percentile(grad[np.isfinite(grad)], 94))
    initial_item = dict(
        center_y=anchor['y'], center_x=anchor['x'], ym=anchor['y'], xm=anchor['x'],
        y1=anchor['y'], x1=anchor['x'], y2=anchor['y'], x2=anchor['x'],
        left_width_px=anchor['left_width'], right_width_px=anchor['right_width'],
        width_px=anchor['left_width'] + anchor['right_width'],
        tangent_angle_rad=anchor['angle'], tangent_angle_deg=float(np.degrees(anchor['angle']) % 180),
        confidence=anchor['confidence'], score=anchor['confidence'], local_score=anchor['confidence'],
        method='block', detector='block', anchor=True, bundle_score=0.0, bundle_votes=0)
    beam = [dict(
        y=anchor['y'], x=anchor['x'], angle=anchor['angle'],
        left_width=float(np.clip(anchor['left_width'], min_half, max_half)),
        right_width=float(np.clip(anchor['right_width'], min_half, max_half)),
        left_trend=float(anchor['left_trend']), right_trend=float(anchor['right_trend']),
        cumulative=float(anchor['confidence']), steps=1, weak_count=0,
        path=[initial_item], avg_edge=0.5, avg_orientation=0.5,
        avg_bundle=0.0, bundle_hits=0)]
    terminals = []
    for _ in range(BLOCK_MAX_STEPS):
        expanded = []
        for state in beam:
            angle_candidates = []
            for turn in BLOCK_TURN_DEGREES:
                angle = state['angle'] + np.deg2rad(turn)
                y = state['y'] + BLOCK_STEP_PX * np.sin(angle)
                x = state['x'] + BLOCK_STEP_PX * np.cos(angle)
                if not (3 <= y < img.shape[0] - 3 and 3 <= x < img.shape[1] - 3):
                    continue
                ori, _ = _directed_orientation_score(sample_map(theta, y, x), sample_map(coh, y, x), angle)
                preliminary = 0.55 * sample_map(ridge_response, y, x) + 0.45 * ori - 0.05 * abs(turn) / 45.0
                angle_candidates.append((preliminary, angle))
            angle_candidates.sort(reverse=True, key=lambda v: v[0])
            for _, angle in angle_candidates[:BLOCK_ANGLE_BRANCHES]:
                left_options = propose_half_widths(state['left_width'], state['left_trend'], min_half, max_half)
                right_options = propose_half_widths(state['right_width'], state['right_trend'], min_half, max_half)
                combos = representative_width_combos(left_options, right_options)
                seen = set()
                for L, R in combos:
                    key = (round(L[0], 2), round(R[0], 2))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = _evaluate_block_candidate(
                        img, ridge_response, theta, coh, gy, gx, grad,
                        dense_dist, continuous_dist, junction_dist,
                        small_ridge_dist, thin_lock_model,
                        state, angle, L[0], R[0], L[1], R[1], L[2], R[2], grad_ref)
                    if candidate is not None:
                        if candidate['weak_count'] > BLOCK_MAX_WEAK_STEPS:
                            terminals.append(candidate)
                        else:
                            expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(key=_state_objective, reverse=True)
        dedup, bins = [], set()
        for state in expanded:
            key = (int(round(state['y'] / 2.0)), int(round(state['x'] / 2.0)),
                   int(round((np.degrees(state['angle']) % 180) / 10.0)),
                   int(round(state['left_width'] / 2.0)), int(round(state['right_width'] / 2.0)))
            if key in bins:
                continue
            bins.add(key)
            dedup.append(state)
            if len(dedup) >= BLOCK_BEAM_WIDTH:
                break
        beam = dedup
        terminals.extend(beam)
    if not terminals:
        return [], dict(track_id=track_id, grade='C', reason='no_track')
    valid = [s for s in terminals if s['steps'] >= 2]
    best = max(valid, key=_state_objective)
    avg_conf = best['cumulative'] / max(best['steps'], 1)
    length = (best['steps'] - 1) * BLOCK_STEP_PX
    bundle_hit_fraction = best.get('bundle_hits', 0) / max(best['steps'] - 1, 1)
    grade = ('B' if best['steps'] >= BLOCK_MIN_TRACK_STEPS and
             length >= BLOCK_MIN_TRACK_LENGTH_PX and
             avg_conf >= BLOCK_GRADE_B_MIN_AVG_CONFIDENCE and
             best['avg_edge'] >= BLOCK_GRADE_B_MIN_EDGE_SUPPORT and
             best['avg_orientation'] >= BLOCK_GRADE_B_MIN_ORIENTATION and
             best.get('avg_bundle', 0.0) <= BUNDLE_BLOCK_MAX_AVG_SCORE and
             bundle_hit_fraction <= BUNDLE_BLOCK_MAX_HIT_FRACTION else 'C')
    path = []
    for i, item in enumerate(best['path'][1::BLOCK_PATH_SAMPLE_STRIDE]):
        d = canonicalize_measurement(item, 'block')
        d['grade'] = grade
        d['track_id'] = track_id
        d['track_step'] = i
        d['anchor_source'] = anchor['source']
        d['track_length_px'] = float(length)
        d['track_avg_confidence'] = float(avg_conf)
        d['track_avg_edge_support'] = float(best['avg_edge'])
        d['track_avg_orientation'] = float(best['avg_orientation'])
        d['track_avg_bundle_score'] = float(best.get('avg_bundle', 0.0))
        d['track_bundle_hit_fraction'] = float(bundle_hit_fraction)
        d['confidence'] = float(np.clip(0.68 * d['local_score'] + 0.32 * avg_conf, 0, 1))
        path.append(d)
    meta = dict(track_id=track_id, grade=grade, anchor_source=anchor['source'],
                n_steps=best['steps'], track_length_px=float(length),
                avg_confidence=float(avg_conf), avg_edge_support=float(best['avg_edge']),
                avg_orientation=float(best['avg_orientation']),
                avg_bundle_score=float(best.get('avg_bundle', 0.0)),
                bundle_hit_fraction=float(bundle_hit_fraction))
    return path, meta


def run_adaptive_block_tracks(img, ridge_response, theta, coh, gy, gx, grad,
                              continuous_edge, dense_edge, junction_labels,
                              small_ridge_dist, thin_lock_model,
                              anchors, max_width):
    dense_dist = ndimage.distance_transform_edt(~dense_edge)
    continuous_dist = ndimage.distance_transform_edt(~continuous_edge)
    junction_dist = (ndimage.distance_transform_edt(~(junction_labels > 0))
                     if junction_labels.max() > 0 else np.full(img.shape, 999.0))
    grade_b, grade_c, metadata = [], [], []
    for track_id, anchor in enumerate(anchors, 1):
        path, meta = run_single_block_track(
            img, ridge_response, theta, coh, gy, gx, grad,
            dense_dist, continuous_dist, junction_dist,
            small_ridge_dist, thin_lock_model,
            anchor, max_width, track_id)
        metadata.append(meta)
        (grade_b if meta['grade'] == 'B' else grade_c).extend(path)
    return grade_b, grade_c, metadata


def draw_consensus_block_result(img, ridge_response, ridge_centerline,
                                small_ridge_response, small_ridge_centerline,
                                thin_lock_model, edge_samples, ridge_samples,
                                bundle_rejected, confirmed, block_b, block_c,
                                final_samples, candidates, name, output_png):
    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    for ax in axes.ravel():
        ax.imshow(img, cmap='gray')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis('off')

    axes[0, 0].imshow(ridge_response, cmap='magma', alpha=0.43)
    axes[0, 0].imshow(np.ma.masked_where(~ridge_centerline, ridge_centerline),
                      cmap='winter', alpha=0.95)
    axes[0, 0].set_title(f'1. broad multi-scale ridge hypotheses\ncenterline px={int(ridge_centerline.sum())}')

    axes[0, 1].imshow(small_ridge_response, cmap='magma', alpha=0.30)
    axes[0, 1].imshow(np.ma.masked_where(~small_ridge_centerline, small_ridge_centerline),
                      cmap='winter', alpha=0.95)
    locked = thin_lock_model.get('samples', [])
    if locked:
        axes[0, 1].scatter([d['center_x'] for d in locked], [d['center_y'] for d in locked],
                           s=3, c='cyan', alpha=0.7, label='locked thin centres')
        axes[0, 1].legend(fontsize=7, loc='upper right')
    axes[0, 1].set_title(f'2. thin-fibre evidence used by bundle guard\nsmall-ridge px={int(small_ridge_centerline.sum())}, locks={len(locked)}')

    add_chords(axes[0, 2], edge_samples, color='cyan', linewidth=0.8, label='edge detector')
    axes[0, 2].set_title(f'3. edge-pair detector\nn={len(edge_samples)}')

    add_chords(axes[0, 3], ridge_samples, color='magenta', linewidth=0.9, label='accepted ridge')
    show_rejected = sorted(bundle_rejected, key=lambda d: d.get('bundle_score', 0), reverse=True)[:900]
    add_chords(axes[0, 3], show_rejected, color='red', linewidth=0.55, label='bundle rejected')
    axes[0, 3].legend(fontsize=7, loc='upper right')
    axes[0, 3].set_title(f'4. ridge detector after bundle guard\naccepted={len(ridge_samples)}, rejected={len(bundle_rejected)}')

    add_chords(axes[1, 0], confirmed, color='yellow', linewidth=1.4, label='Grade A')
    axes[1, 0].set_title(f'5. detector consensus anchors\nGrade A n={len(confirmed)}')

    _draw_center_paths(axes[1, 1], block_b, 'lime', 'Grade B track', 1.4)
    _draw_center_paths(axes[1, 1], block_c, 'orange', 'Grade C track', 0.7)
    if block_b or block_c:
        axes[1, 1].legend(fontsize=7, loc='upper right')
    axes[1, 1].set_title(f'6. bundle-aware ribbon-block tracks\nB={len(block_b)}, C={len(block_c)}')

    grade_a = [d for d in final_samples if d.get('grade') == 'A']
    grade_b = [d for d in final_samples if d.get('grade') == 'B']
    add_chords(axes[1, 2], grade_a, color='yellow', linewidth=1.3, label='A')
    add_chords(axes[1, 2], grade_b, color='lime', linewidth=1.0, label='B')
    show_c = sorted(candidates, key=lambda d: d['confidence'], reverse=True)[:450]
    add_chords(axes[1, 2], show_c, color='orange', linewidth=0.35, label='C candidate')
    if grade_a or grade_b or show_c:
        axes[1, 2].legend(fontsize=7, loc='upper right')
    median = np.median([d['width_px'] for d in final_samples]) if final_samples else np.nan
    axes[1, 2].set_title(f'7. final thickness and candidates\naccepted={len(final_samples)}, median={median:.2f}px')

    axes[1, 3].clear()
    axes[1, 3].axis('on')
    axes[1, 3].set_aspect('auto')
    for grade, color in (('A', 'gold'), ('B', 'tab:green')):
        vals = [d['width_px'] for d in final_samples if d.get('grade') == grade]
        if vals:
            axes[1, 3].hist(vals, bins='auto', alpha=0.58, color=color,
                            label=f'{grade} (n={len(vals)})')
    axes[1, 3].set_xlabel('thickness (px)')
    axes[1, 3].set_ylabel('accepted measurements')
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].set_title('8. final thickness distribution')

    fig.suptitle(f'{name} | ridge-edge consensus + bundle guard + adaptive blocks', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches='tight')
    if SHOW_INLINE:
        plt.show()
    plt.close(fig)


def process_one(path):
    img, cut = prepare_image(path)
    gy, gx, grad, theta, coh, energy = orientation_fields(img)
    continuous_edge = build_continuous_edges(img, coh, energy)
    dense_edge, dense_threshold = build_dense_edge_candidates(grad, gy, gx, coh, energy)
    edge_curves, edge_junctions = trace_edge_curves(continuous_edge)

    thin_samples, thin_rejected, thin_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'thin')
    thick_samples, thick_rejected, thick_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'thick')
    curved_samples, curved_rejected, curved_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'curved')
    edge_samples = [canonicalize_measurement(v, 'edge') for v in
                    merge_method_results(thin_samples, thick_samples, curved_samples)]
    thin_lock_model = build_thin_lock_model(edge_samples)

    (ridge_response, ridge_centerline, ridge_curves, ridge_junctions,
     small_ridge_response, small_ridge_centerline, small_ridge_curves,
     small_ridge_junctions) = build_bright_ridge_centerlines(img)
    small_ridge_dist = ndimage.distance_transform_edt(~small_ridge_centerline)

    ridge_samples, ridge_rejected, ridge_max, bundle_rejected = measure_ridge_detector(
        img, ridge_response, ridge_curves, gy, gx, grad, theta, coh,
        continuous_edge, dense_edge, small_ridge_dist, thin_lock_model)

    confirmed, ridge_only, edge_only = match_detector_consensus(ridge_samples, edge_samples)
    anchors = build_tracking_anchors(confirmed, ridge_only, edge_only)
    block_b, block_c, track_meta = run_adaptive_block_tracks(
        img, ridge_response, theta, coh, gy, gx, grad,
        continuous_edge, dense_edge, edge_junctions,
        small_ridge_dist, thin_lock_model,
        anchors, max(thin_max, thick_max, curved_max, ridge_max))
    final_samples, candidates = merge_final_measurements(
        confirmed, block_b, ridge_only + edge_only + block_c)

    final_df = pd.DataFrame(final_samples)
    if len(final_df):
        final_df.insert(0, 'file', path.name)
        final_df['status'] = 'accepted'
        nm_px = NM_PER_PX.get(path.name)
        final_df['nm_per_px'] = nm_px if nm_px is not None else np.nan
        final_df['thickness_nm'] = final_df['width_px'] * nm_px if nm_px is not None else np.nan
    candidate_df = pd.DataFrame(candidates)
    if len(candidate_df):
        candidate_df.insert(0, 'file', path.name)
        candidate_df['status'] = 'candidate_only'
    detector_df = pd.DataFrame(edge_samples + ridge_samples)
    if len(detector_df):
        detector_df.insert(0, 'file', path.name)
    rejection_df = pd.DataFrame(thin_rejected + thick_rejected + curved_rejected + ridge_rejected)
    if len(rejection_df):
        rejection_df.insert(0, 'file', path.name)
    track_df = pd.DataFrame(track_meta)
    if len(track_df):
        track_df.insert(0, 'file', path.name)
    bundle_df = pd.DataFrame(bundle_rejected)
    if len(bundle_df):
        bundle_df.insert(0, 'file', path.name)
        bundle_df['status'] = 'bundle_rejected'

    output_png = OUTPUT_DIR / f'{path.stem}_bundle_guard_thickness.png'
    draw_consensus_block_result(
        img, ridge_response, ridge_centerline,
        small_ridge_response, small_ridge_centerline,
        thin_lock_model, edge_samples, ridge_samples, bundle_rejected,
        confirmed, block_b, block_c, final_samples, candidates,
        path.name, output_png)

    final_df.to_csv(OUTPUT_DIR / f'{path.stem}_accepted_A_B_thickness.csv', index=False)
    candidate_df.to_csv(OUTPUT_DIR / f'{path.stem}_grade_C_candidates.csv', index=False)
    detector_df.to_csv(OUTPUT_DIR / f'{path.stem}_raw_detector_measurements.csv', index=False)
    track_df.to_csv(OUTPUT_DIR / f'{path.stem}_block_track_summary.csv', index=False)
    rejection_df.to_csv(OUTPUT_DIR / f'{path.stem}_rejected_runs.csv', index=False)
    bundle_df.to_csv(OUTPUT_DIR / f'{path.stem}_bundle_rejected_measurements.csv', index=False)

    widths = final_df['width_px'].to_numpy(float) if len(final_df) else np.array([])
    grades = final_df['grade'].value_counts().to_dict() if len(final_df) else {}
    summary = dict(
        file=path.name, cropped_height=cut, image_height=img.shape[0], image_width=img.shape[1],
        edge_detector_measurements=len(edge_samples), ridge_detector_measurements=len(ridge_samples),
        locked_thin_measurements=len(thin_lock_model['samples']),
        bundle_rejected_measurements=len(bundle_df),
        bundle_rejected_runs=sum(str(r.get('reason', '')).startswith('multiple_thin_fibres')
                                 for r in ridge_rejected),
        grade_A_measurements=int(grades.get('A', 0)),
        grade_B_measurements=int(grades.get('B', 0)),
        grade_C_candidates=len(candidate_df), block_anchors=len(anchors),
        grade_B_tracks=sum(m.get('grade') == 'B' for m in track_meta),
        grade_C_tracks=sum(m.get('grade') == 'C' for m in track_meta),
        accepted_measurements=len(final_df),
        median_thickness_px=float(np.median(widths)) if len(widths) else np.nan,
        p25_thickness_px=float(np.percentile(widths, 25)) if len(widths) else np.nan,
        p75_thickness_px=float(np.percentile(widths, 75)) if len(widths) else np.nan,
        max_thickness_px=float(np.max(widths)) if len(widths) else np.nan,
        dense_gradient_threshold=dense_threshold)
    return summary, final_df, candidate_df


# ============================================================================
# SEM-agreement + fibre-region extension
#
# The local detectors above intentionally produce many thickness chords. This
# extension (1) refines each chord against the original SEM image, (2) groups
# neighbouring chords into one visible fibre region, and (3) emits one or more
# representative widths per region instead of counting every local chord as an
# independent fibre. A region is a continuous, confidently measured part of a
# fibre, not necessarily the physical full fibre from tip to tip.
# ============================================================================
TARGET_FILES = ['2-10.jpg', '2-11.jpg', '2-19.jpg']
OUTPUT_DIR = IMAGE_DIR.parent / f'{IMAGE_DIR.name}_fiber_regions_sem_refined'

# Local SEM-image refinement. Small centre/angle/width adjustments are tested
# against the raw intensity, gradient, orientation, ridge and bundle evidence.
SEM_REFINE_ANGLE_DEG = (-5.0, 0.0, 5.0)
SEM_REFINE_CENTER_SHIFT_PX = (-1.0, 0.0, 1.0)
SEM_REFINE_WIDTH_DELTA_PX = (-1.25, 0.0, 1.25)
SEM_REFINE_MAX_REL_WIDTH_CHANGE = 0.30
SEM_LOCAL_MIN_AGREEMENT_A = 0.29
SEM_LOCAL_MIN_AGREEMENT_B = 0.34
SEM_LOCAL_STRONG_BUNDLE_VOTES = 2
SEM_LOCAL_STRONG_BUNDLE_SCORE = 0.66

# Centreline assignment and region construction.
REGION_SMALL_CURVE_MAX_WIDTH_PX = 14.0
REGION_SMALL_CURVE_ASSIGN_PX = 3.0
REGION_BROAD_CURVE_ASSIGN_PX = 4.5
REGION_LINK_RADIUS_PX = 9.0
REGION_LINK_MAX_TANGENT_DIFF_DEG = 27.0
REGION_LINK_MAX_NORMAL_OFFSET_PX = 2.6
REGION_LINK_NORMAL_WIDTH_FRAC = 0.24
REGION_LINK_MAX_WIDTH_REL_DIFF = 0.48
REGION_LINK_MAX_BOUNDARY_EXCESS_PX = 4.5
REGION_SPLIT_MAX_ARC_GAP_PX = 12.0
REGION_SPLIT_MAX_WIDTH_JUMP_FRAC = 0.58
REGION_MIN_SAMPLES = 3
REGION_MIN_GRADE_A_SAMPLES = 2
REGION_MIN_SPAN_PX = 5.0
REGION_MIN_MEDIAN_SEM_AGREEMENT = 0.31
REGION_MIN_MEDIAN_CONFIDENCE = 0.38
REGION_OUTLIER_MIN_RESIDUAL_PX = 1.4
REGION_OUTLIER_MAD_FACTOR = 3.5

# One representative width is emitted for nearly uniform regions. Tapered or
# multi-width regions may receive 2-3 contiguous representative subregions.
REGION_SPLIT_REL_RANGE = 0.30
REGION_SPLIT_MIN_SAMPLES = 10
REGION_MAX_SUBREGIONS = 3
REGION_MIN_SUBREGION_SAMPLES = 4
REGION_SEGMENT_PENALTY_FRAC = 0.11
REGION_MIN_REP_DIFFERENCE_FRAC = 0.14
REGION_MIN_SEGMENT_IMPROVEMENT = 0.22

# Diagnostics.
REGION_LABEL_LIMIT = 80
REGION_CANDIDATE_DRAW_LIMIT = 700


def weighted_histogram_bin_count(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return max(1, min(30, max(6, int(np.sqrt(max(len(finite), 1)) * 2))))


def _weighted_median(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return np.nan
    values, weights = values[good], weights[good]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    c = np.cumsum(weights)
    return float(values[np.searchsorted(c, 0.5 * c[-1], side='left')])


def _measurement_geometry(item):
    d = canonicalize_measurement(item)
    angle = float(d['tangent_angle_rad'])
    ty, tx = float(np.sin(angle)), float(np.cos(angle))
    ny, nx = tx, -ty
    y, x = d['center_y'], d['center_x']
    o1 = (d['y1'] - y) * ny + (d['x1'] - x) * nx
    o2 = (d['y2'] - y) * ny + (d['x2'] - x) * nx
    left = max(0.75, -min(o1, o2))
    right = max(0.75, max(o1, o2))
    # Fall back to symmetric halves when endpoint orientation is inconsistent.
    if left + right < 0.55 * d['width_px']:
        left = right = max(0.75, 0.5 * d['width_px'])
    return d, angle, ty, tx, ny, nx, float(left), float(right)


def _geometry_to_measurement(base, y, x, angle, left_width, right_width):
    ty, tx = float(np.sin(angle)), float(np.cos(angle))
    ny, nx = tx, -ty
    d = dict(base)
    d.update(
        center_y=float(y), center_x=float(x), ym=float(y), xm=float(x),
        y1=float(y - ny * left_width), x1=float(x - nx * left_width),
        y2=float(y + ny * right_width), x2=float(x + nx * right_width),
        left_width_px=float(left_width), right_width_px=float(right_width),
        width_px=float(left_width + right_width),
        tangent_angle_rad=float(angle),
        tangent_angle_deg=float(np.degrees(angle) % 180.0),
    )
    return d


def sem_ribbon_agreement(img, ridge_response, theta, coh, gy, gx, grad,
                         continuous_dist, dense_dist, small_ridge_dist,
                         thin_lock_model, item, grad_ref):
    """Score how well one predicted ribbon agrees with the original SEM image."""
    d, angle, ty, tx, ny, nx, left, right = _measurement_geometry(item)
    y, x = d['center_y'], d['center_x']
    ly, lx, ry, rx = d['y1'], d['x1'], d['y2'], d['x2']
    h, w = img.shape
    if not (2 <= y < h - 2 and 2 <= x < w - 2 and
            1 <= ly < h - 1 and 1 <= lx < w - 1 and
            1 <= ry < h - 1 and 1 <= rx < w - 1):
        return dict(sem_agreement=0.0, sem_valid=False, reason='outside_image')

    ld = min(sample_map(continuous_dist, ly, lx), sample_map(dense_dist, ly, lx))
    rd = min(sample_map(continuous_dist, ry, rx), sample_map(dense_dist, ry, rx))
    left_support = float(np.exp(-0.5 * (ld / 1.8) ** 2))
    right_support = float(np.exp(-0.5 * (rd / 1.8) ** 2))
    edge_support = 0.5 * (left_support + right_support)

    lgy, lgx = sample_map(gy, ly, lx), sample_map(gx, ly, lx)
    rgy, rgx = sample_map(gy, ry, rx), sample_map(gx, ry, rx)
    # At the two outer boundaries, projections along opposite outward normals
    # should both descend from the bright ribbon into the background.
    dl = lgy * (-ny) + lgx * (-nx)
    dr = rgy * ny + rgx * nx
    edge_strength = float(np.clip(0.5 * (abs(dl) + abs(dr)) / max(grad_ref, 1e-9), 0, 1))
    polarity_score = 1.0 if dl < 0 and dr < 0 else (0.55 if dl * dr > 0 else 0.25)

    ts = np.linspace(-left + 0.6, right - 0.6,
                     max(7, int(round((left + right) / 1.2))))
    interior = _sample_line(img, y, x, ny, nx, ts)
    theta_in = _sample_line(theta, y, x, ny, nx, ts)
    coh_in = _sample_line(coh, y, x, ny, nx, ts)
    theta_ref = float(np.degrees(angle) % 180.0)
    orientation_score = orientation_profile_score(
        theta_in, coh_in, theta_ref, min_coherency=0.06, tolerance_deg=38.0)

    outside_l = _sample_line(img, y, x, ny, nx,
                             np.array([-left - 3.0, -left - 2.0, -left - 1.2], np.float32))
    outside_r = _sample_line(img, y, x, ny, nx,
                             np.array([right + 1.2, right + 2.0, right + 3.0], np.float32))
    inside_level = float(np.median(interior))
    outside_level = float(np.median(np.r_[outside_l, outside_r]))
    contrast = inside_level - outside_level
    contrast_score = float(np.clip((contrast + 0.015) / 0.13, 0, 1))
    ridge_score = float(sample_map(ridge_response, y, x))

    bundle = bundle_evidence_at_ribbon(
        img, small_ridge_dist, thin_lock_model, y, x, ty, tx, left, right)
    bundle_penalty = float(bundle['bundle_score'])

    prior_conf = float(np.clip(d.get('confidence', 0.0), 0, 1))
    agreement = (0.25 * edge_support + 0.18 * edge_strength +
                 0.10 * polarity_score + 0.17 * orientation_score +
                 0.12 * ridge_score + 0.10 * contrast_score +
                 0.08 * prior_conf)
    agreement -= 0.24 * bundle_penalty
    if bundle['bundle_votes'] >= 2:
        agreement -= 0.10
    agreement = float(np.clip(agreement, 0, 1))
    return dict(
        sem_agreement=agreement,
        sem_valid=True,
        sem_edge_support=float(edge_support),
        sem_edge_strength=float(edge_strength),
        sem_polarity_score=float(polarity_score),
        sem_orientation_score=float(orientation_score),
        sem_ridge_score=float(ridge_score),
        sem_contrast=float(contrast),
        sem_contrast_score=float(contrast_score),
        **bundle,
    )


def refine_measurements_against_sem(img, ridge_response, theta, coh, gy, gx, grad,
                                    continuous_edge, dense_edge, small_ridge_dist,
                                    thin_lock_model, measurements):
    """Locally adjust centre, tangent and left/right widths to fit the SEM image."""
    continuous_dist = ndimage.distance_transform_edt(~continuous_edge)
    dense_dist = ndimage.distance_transform_edt(~dense_edge)
    grad_ref = float(np.percentile(grad[np.isfinite(grad)], 94))
    refined, downgraded = [], []
    for raw in measurements:
        base, angle0, ty0, tx0, ny0, nx0, left0, right0 = _measurement_geometry(raw)
        best, best_metrics, best_objective = None, None, -np.inf
        for angle_delta in SEM_REFINE_ANGLE_DEG:
            angle = angle0 + np.deg2rad(angle_delta)
            ty, tx = float(np.sin(angle)), float(np.cos(angle))
            ny, nx = tx, -ty
            for shift in SEM_REFINE_CENTER_SHIFT_PX:
                y = base['center_y'] + shift * ny
                x = base['center_x'] + shift * nx
                for dl in SEM_REFINE_WIDTH_DELTA_PX:
                    for dr in SEM_REFINE_WIDTH_DELTA_PX:
                        left = max(0.9, left0 + dl)
                        right = max(0.9, right0 + dr)
                        width = left + right
                        rel = abs(width - base['width_px']) / max(base['width_px'], 1e-6)
                        if rel > SEM_REFINE_MAX_REL_WIDTH_CHANGE:
                            continue
                        cand = _geometry_to_measurement(base, y, x, angle, left, right)
                        metrics = sem_ribbon_agreement(
                            img, ridge_response, theta, coh, gy, gx, grad,
                            continuous_dist, dense_dist, small_ridge_dist,
                            thin_lock_model, cand, grad_ref)
                        move_penalty = (0.020 * abs(shift) +
                                        0.012 * abs(angle_delta) / 5.0 +
                                        0.020 * (abs(dl) + abs(dr)) / 1.25)
                        objective = metrics['sem_agreement'] - move_penalty
                        if objective > best_objective:
                            best, best_metrics, best_objective = cand, metrics, objective
        if best is None:
            d = canonicalize_measurement(base)
            d.update(sem_agreement=0.0, sem_valid=False, sem_refined=False,
                     sem_reject_reason='no_valid_refinement')
            downgraded.append(d)
            continue
        d = canonicalize_measurement(best)
        d.update(best_metrics)
        d['sem_refined'] = True
        d['original_center_y'] = float(base['center_y'])
        d['original_center_x'] = float(base['center_x'])
        d['original_width_px'] = float(base['width_px'])
        d['sem_center_shift_px'] = float(np.hypot(d['center_y'] - base['center_y'],
                                                  d['center_x'] - base['center_x']))
        d['sem_width_change_px'] = float(d['width_px'] - base['width_px'])
        grade = str(d.get('grade', 'C'))
        threshold = SEM_LOCAL_MIN_AGREEMENT_A if grade == 'A' else SEM_LOCAL_MIN_AGREEMENT_B
        strong_bundle = (d.get('bundle_votes', 0) >= SEM_LOCAL_STRONG_BUNDLE_VOTES and
                         d.get('bundle_score', 0.0) >= SEM_LOCAL_STRONG_BUNDLE_SCORE)
        d['sem_local_keep'] = bool(d['sem_agreement'] >= threshold and not strong_bundle)
        d['confidence'] = float(np.clip(0.46 * d.get('confidence', 0.0) +
                                        0.54 * d['sem_agreement'], 0, 1))
        d['score'] = d['confidence']
        if d['sem_local_keep']:
            refined.append(d)
        else:
            d['sem_reject_reason'] = ('strong_bundle' if strong_bundle else
                                      'low_sem_agreement')
            downgraded.append(d)
    return refined, downgraded


def build_curve_index(curves):
    points, curve_ids, arc_values = [], [], []
    for cid, curve in enumerate(curves):
        p, arc = resample_polyline(np.asarray(curve, float), ds=1.0)
        if len(p) < 2:
            continue
        points.extend(p.tolist())
        curve_ids.extend([cid] * len(p))
        arc_values.extend(np.asarray(arc, float).tolist())
    if not points:
        return dict(tree=None, points=np.empty((0, 2)), curve_ids=np.array([], int),
                    arc=np.array([], float))
    pts = np.asarray(points, float)
    return dict(tree=cKDTree(pts), points=pts,
                curve_ids=np.asarray(curve_ids, int), arc=np.asarray(arc_values, float))


def _query_curve_index(index, y, x, max_distance):
    tree = index.get('tree') if index else None
    if tree is None:
        return None
    dist, idx = tree.query([y, x], k=1)
    if not np.isfinite(dist) or dist > max_distance:
        return None
    return dict(distance=float(dist), curve_id=int(index['curve_ids'][idx]),
                arc_s=float(index['arc'][idx]), point_y=float(index['points'][idx, 0]),
                point_x=float(index['points'][idx, 1]))


def annotate_region_basis(samples, broad_curves, small_curves):
    broad_index = build_curve_index(broad_curves)
    small_index = build_curve_index(small_curves)
    out = []
    for raw in samples:
        d = canonicalize_measurement(raw)
        small = _query_curve_index(small_index, d['center_y'], d['center_x'],
                                   REGION_SMALL_CURVE_ASSIGN_PX)
        broad = _query_curve_index(broad_index, d['center_y'], d['center_x'],
                                   REGION_BROAD_CURVE_ASSIGN_PX)
        use_small = (
            d['width_px'] <= REGION_SMALL_CURVE_MAX_WIDTH_PX and
            small is not None and
            (broad is None or
             small['distance'] + 0.75 < broad['distance'] or
             d.get('bundle_score', 0.0) >= 0.22)
        )
        if use_small:
            d['region_basis'] = 'small_ridge'
            d['region_curve_id'] = small['curve_id']
            d['region_order_s'] = small['arc_s']
            d['region_curve_distance_px'] = small['distance']
        elif broad is not None:
            d['region_basis'] = 'broad_ridge'
            d['region_curve_id'] = broad['curve_id']
            d['region_order_s'] = broad['arc_s']
            d['region_curve_distance_px'] = broad['distance']
        elif d.get('detector') == 'block' and int(d.get('track_id', -1)) >= 0:
            d['region_basis'] = 'block_track'
            d['region_curve_id'] = int(d.get('track_id', -1))
            d['region_order_s'] = float(d.get('track_step', 0)) * BLOCK_STEP_PX
            d['region_curve_distance_px'] = 0.0
        else:
            d['region_basis'] = 'fallback'
            d['region_curve_id'] = -1
            d['region_order_s'] = np.nan
            d['region_curve_distance_px'] = np.nan
        out.append(d)
    return out


def _boundary_pair_distance(a, b):
    p1 = np.array([[a['y1'], a['x1']], [a['y2'], a['x2']]], float)
    p2 = np.array([[b['y1'], b['x1']], [b['y2'], b['x2']]], float)
    direct = 0.5 * (np.linalg.norm(p1[0] - p2[0]) + np.linalg.norm(p1[1] - p2[1]))
    cross = 0.5 * (np.linalg.norm(p1[0] - p2[1]) + np.linalg.norm(p1[1] - p2[0]))
    return float(min(direct, cross))


def region_link_score(a, b):
    """Return a continuity cost, or None when two chords should not share a region."""
    a, b = canonicalize_measurement(a), canonicalize_measurement(b)
    dy, dx = b['center_y'] - a['center_y'], b['center_x'] - a['center_x']
    distance = float(np.hypot(dy, dx))
    if not (0.35 <= distance <= REGION_LINK_RADIUS_PX):
        return None
    angle_a, angle_b = tangent_angle_from_item(a), tangent_angle_from_item(b)
    diff = axial_angle_diff_deg(np.degrees(angle_a), np.degrees(angle_b))
    if diff > REGION_LINK_MAX_TANGENT_DIFF_DEG:
        return None
    # Axial tangents have no sign; choose the sign that best aligns with delta.
    ta = np.array([np.sin(angle_a), np.cos(angle_a)], float)
    if np.dot(ta, [dy, dx]) < 0:
        ta *= -1
    na = np.array([ta[1], -ta[0]], float)
    normal_offset = abs(float(np.dot([dy, dx], na)))
    allowed_normal = max(REGION_LINK_MAX_NORMAL_OFFSET_PX,
                         REGION_LINK_NORMAL_WIDTH_FRAC * min(a['width_px'], b['width_px']))
    if normal_offset > allowed_normal:
        return None
    width_rel = abs(a['width_px'] - b['width_px']) / max(0.5 * (a['width_px'] + b['width_px']), 1e-6)
    if width_rel > REGION_LINK_MAX_WIDTH_REL_DIFF:
        return None
    boundary_distance = _boundary_pair_distance(a, b)
    if boundary_distance > distance + REGION_LINK_MAX_BOUNDARY_EXCESS_PX:
        return None
    sem_penalty = 1.0 - 0.5 * (a.get('sem_agreement', 0.5) + b.get('sem_agreement', 0.5))
    return float(distance + 1.3 * normal_offset + 2.0 * width_rel +
                 0.6 * diff / 10.0 + 0.25 * boundary_distance + sem_penalty)


def _fallback_components(samples):
    if not samples:
        return []
    centers = np.array([[d['center_y'], d['center_x']] for d in samples], float)
    tree = cKDTree(centers)
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, d in enumerate(samples):
        candidates = tree.query_ball_point(centers[i], REGION_LINK_RADIUS_PX)
        scored = []
        for j in candidates:
            if j <= i:
                continue
            score = region_link_score(d, samples[j])
            if score is not None:
                scored.append((score, j))
        for _, j in sorted(scored)[:2]:
            union(i, j)
    groups = {}
    for i, d in enumerate(samples):
        groups.setdefault(find(i), []).append(d)
    return list(groups.values())


def _order_fallback_sequence(seq):
    if len(seq) <= 2:
        return seq
    centers = np.array([[d['center_y'], d['center_x']] for d in seq], float)
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    order = np.argsort(centered @ axis)
    return [seq[i] for i in order]


def _split_ordered_sequence(seq):
    if not seq:
        return []
    pieces, current = [], [seq[0]]
    for prev, cur in zip(seq[:-1], seq[1:]):
        gap = float(np.hypot(cur['center_y'] - prev['center_y'],
                             cur['center_x'] - prev['center_x']))
        width_jump = abs(cur['width_px'] - prev['width_px']) / max(
            0.5 * (cur['width_px'] + prev['width_px']), 1e-6)
        link = region_link_score(prev, cur)
        if (gap > REGION_SPLIT_MAX_ARC_GAP_PX or
                width_jump > REGION_SPLIT_MAX_WIDTH_JUMP_FRAC or link is None):
            pieces.append(current)
            current = [cur]
        else:
            current.append(cur)
    pieces.append(current)
    return pieces


def build_fiber_regions(samples, broad_curves, small_curves):
    annotated = annotate_region_basis(samples, broad_curves, small_curves)
    keyed, fallback = {}, []
    for d in annotated:
        if d['region_basis'] == 'fallback':
            fallback.append(d)
        else:
            key = (d['region_basis'], int(d['region_curve_id']))
            keyed.setdefault(key, []).append(d)

    provisional = []
    for key, seq in keyed.items():
        seq.sort(key=lambda d: d.get('region_order_s', 0.0))
        for piece in _split_ordered_sequence(seq):
            provisional.append((key, piece))
    for seq in _fallback_components(fallback):
        seq = _order_fallback_sequence(seq)
        for piece in _split_ordered_sequence(seq):
            provisional.append((('fallback', -1), piece))

    regions = []
    region_id = 1
    for (basis, curve_id), seq in provisional:
        if not seq:
            continue
        centers = np.array([[d['center_y'], d['center_x']] for d in seq], float)
        span = float(np.sum(np.hypot(np.diff(centers[:, 0]), np.diff(centers[:, 1])))) if len(seq) > 1 else 0.0
        grade_a_count = sum(str(d.get('grade', '')) == 'A' for d in seq)
        min_samples = REGION_MIN_GRADE_A_SAMPLES if grade_a_count >= REGION_MIN_GRADE_A_SAMPLES else REGION_MIN_SAMPLES
        if len(seq) < min_samples or span < REGION_MIN_SPAN_PX:
            continue
        sem_med = float(np.median([d.get('sem_agreement', 0.0) for d in seq]))
        conf_med = float(np.median([d.get('confidence', 0.0) for d in seq]))
        if sem_med < REGION_MIN_MEDIAN_SEM_AGREEMENT and conf_med < REGION_MIN_MEDIAN_CONFIDENCE:
            continue

        # Robustly remove isolated local-width failures without flattening gradual taper.
        widths = np.asarray([d['width_px'] for d in seq], float)
        if len(widths) >= 5:
            kernel = min(7, len(widths) if len(widths) % 2 == 1 else len(widths) - 1)
            trend = ndimage.median_filter(widths, size=max(3, kernel), mode='nearest')
            residual = np.abs(widths - trend)
            mad = float(np.median(np.abs(residual - np.median(residual))))
            threshold = max(REGION_OUTLIER_MIN_RESIDUAL_PX,
                            REGION_OUTLIER_MAD_FACTOR * 1.4826 * mad)
            kept = [d for d, r in zip(seq, residual) if r <= threshold]
            if len(kept) >= min_samples:
                seq = kept
                centers = np.array([[d['center_y'], d['center_x']] for d in seq], float)
                span = float(np.sum(np.hypot(np.diff(centers[:, 0]), np.diff(centers[:, 1])))) if len(seq) > 1 else 0.0

        for i, d in enumerate(seq):
            d['fiber_region_id'] = region_id
            d['region_sample_index'] = i
        regions.append(dict(
            fiber_region_id=region_id,
            region_basis=basis,
            region_curve_id=int(curve_id),
            samples=seq,
            n_samples=len(seq),
            region_span_px=span,
            start_y=float(seq[0]['center_y']), start_x=float(seq[0]['center_x']),
            end_y=float(seq[-1]['center_y']), end_x=float(seq[-1]['center_x']),
            median_sem_agreement=float(np.median([d.get('sem_agreement', 0.0) for d in seq])),
            median_confidence=float(np.median([d.get('confidence', 0.0) for d in seq])),
            grade_A_fraction=float(np.mean([str(d.get('grade', '')) == 'A' for d in seq])),
        ))
        region_id += 1
    return regions


def _segment_cost(widths, i, j):
    vals = widths[i:j]
    if len(vals) == 0:
        return np.inf
    med = float(np.median(vals))
    return float(np.sum((vals - med) ** 2))


def segment_region_widths(widths, arc_s):
    """Return 1-3 contiguous representative widths for one fibre region."""
    widths = np.asarray(widths, float)
    arc_s = np.asarray(arc_s, float)
    good = np.isfinite(widths) & np.isfinite(arc_s)
    widths, arc_s = widths[good], arc_s[good]
    n = len(widths)
    if n == 0:
        return []
    order = np.argsort(arc_s)
    widths, arc_s = widths[order], arc_s[order]
    p10, p50, p90 = np.percentile(widths, [10, 50, 90])
    rel_range = float((p90 - p10) / max(p50, 1e-6))

    chosen = [(0, n)]
    if n >= REGION_SPLIT_MIN_SAMPLES and rel_range >= REGION_SPLIT_REL_RANGE:
        one_cost = _segment_cost(widths, 0, n)
        if one_cost > 1e-9:
            best_total = one_cost
            best_segments = chosen
            max_k = min(REGION_MAX_SUBREGIONS, n // REGION_MIN_SUBREGION_SAMPLES)
            for k in range(2, max_k + 1):
                dp = np.full((k + 1, n + 1), np.inf)
                prev = np.full((k + 1, n + 1), -1, int)
                dp[0, 0] = 0.0
                for kk in range(1, k + 1):
                    lo_j = kk * REGION_MIN_SUBREGION_SAMPLES
                    for j in range(lo_j, n + 1):
                        i_min = (kk - 1) * REGION_MIN_SUBREGION_SAMPLES
                        i_max = j - REGION_MIN_SUBREGION_SAMPLES
                        for i in range(i_min, i_max + 1):
                            value = dp[kk - 1, i] + _segment_cost(widths, i, j)
                            if value < dp[kk, j]:
                                dp[kk, j], prev[kk, j] = value, i
                raw_cost = dp[k, n]
                if not np.isfinite(raw_cost):
                    continue
                penalized = raw_cost + (k - 1) * REGION_SEGMENT_PENALTY_FRAC * one_cost
                cuts, j = [], n
                for kk in range(k, 0, -1):
                    i = int(prev[kk, j])
                    if i < 0:
                        cuts = []
                        break
                    cuts.append((i, j))
                    j = i
                cuts.reverse()
                if not cuts:
                    continue
                reps = [float(np.median(widths[i:j])) for i, j in cuts]
                distinct = all(abs(a - b) >= REGION_MIN_REP_DIFFERENCE_FRAC * max(0.5 * (a + b), 1e-6)
                               for a, b in zip(reps[:-1], reps[1:]))
                improvement = (one_cost - raw_cost) / one_cost
                if (distinct and improvement >= REGION_MIN_SEGMENT_IMPROVEMENT and
                        penalized < best_total):
                    best_total, best_segments = penalized, cuts
            chosen = best_segments

    # Estimate segment length weights. Each region contributes a total fibre-count
    # weight of one, irrespective of how many local thickness samples it contains.
    if n == 1:
        point_weights = np.array([1.0])
    else:
        ds = np.diff(arc_s)
        positive = ds[ds > 1e-6]
        fallback = float(np.median(positive)) if len(positive) else 1.0
        ds = np.where(ds > 1e-6, ds, fallback)
        point_weights = np.r_[0.5 * ds[0], 0.5 * (ds[:-1] + ds[1:]), 0.5 * ds[-1]]
    total_length_weight = float(np.sum(point_weights))
    out = []
    for sid, (i, j) in enumerate(chosen, 1):
        vals = widths[i:j]
        weight = float(np.sum(point_weights[i:j]) / max(total_length_weight, 1e-9))
        out.append(dict(
            subregion_id=sid,
            start_index=int(i), end_index=int(j - 1),
            start_arc_s=float(arc_s[i]), end_arc_s=float(arc_s[j - 1]),
            representative_width_px=float(np.median(vals)),
            mean_width_px=float(np.mean(vals)),
            min_width_px=float(np.min(vals)), max_width_px=float(np.max(vals)),
            p10_width_px=float(np.percentile(vals, 10)),
            p90_width_px=float(np.percentile(vals, 90)),
            n_local_samples=int(len(vals)),
            fiber_count_weight=weight,
            region_length_fraction=weight,
        ))
    # Numerical protection: force exact unit total for each fibre region.
    norm = sum(v['fiber_count_weight'] for v in out)
    if norm > 0:
        for v in out:
            v['fiber_count_weight'] /= norm
            v['region_length_fraction'] = v['fiber_count_weight']
    return out


def summarize_fiber_regions(regions, nm_per_px=None):
    region_rows, representative_rows, local_rows = [], [], []
    for region in regions:
        seq = region['samples']
        centers = np.array([[d['center_y'], d['center_x']] for d in seq], float)
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(centers[:, 0]), np.diff(centers[:, 1])))] if len(seq) > 1 else np.array([0.0])
        widths = np.asarray([d['width_px'] for d in seq], float)
        reps = segment_region_widths(widths, arc)
        rid = int(region['fiber_region_id'])
        for d, s in zip(seq, arc):
            row = dict(d)
            row['fiber_region_id'] = rid
            row['region_arc_s'] = float(s)
            local_rows.append(row)
        region_row = {k: v for k, v in region.items() if k != 'samples'}
        region_row.update(
            median_width_px=float(np.median(widths)),
            p10_width_px=float(np.percentile(widths, 10)),
            p90_width_px=float(np.percentile(widths, 90)),
            min_width_px=float(np.min(widths)), max_width_px=float(np.max(widths)),
            n_representative_widths=len(reps),
        )
        if nm_per_px is not None:
            for key in ('median_width_px', 'p10_width_px', 'p90_width_px',
                        'min_width_px', 'max_width_px'):
                region_row[key.replace('_px', '_nm')] = region_row[key] * nm_per_px
        region_rows.append(region_row)
        for rep in reps:
            i, j = rep['start_index'], rep['end_index']
            mid = (i + j) // 2
            sample = seq[mid]
            row = dict(rep)
            row.update(
                fiber_region_id=rid,
                region_basis=region['region_basis'],
                region_curve_id=region['region_curve_id'],
                region_span_px=region['region_span_px'],
                region_median_sem_agreement=region['median_sem_agreement'],
                region_median_confidence=region['median_confidence'],
                center_y=float(sample['center_y']), center_x=float(sample['center_x']),
                y1=float(sample['y1']), x1=float(sample['x1']),
                y2=float(sample['y2']), x2=float(sample['x2']),
                tangent_angle_deg=float(sample.get('tangent_angle_deg', 0.0)),
                length_weight=float(region['region_span_px'] * rep['region_length_fraction']),
            )
            if nm_per_px is not None:
                row['representative_width_nm'] = row['representative_width_px'] * nm_per_px
            representative_rows.append(row)
    return pd.DataFrame(region_rows), pd.DataFrame(representative_rows), pd.DataFrame(local_rows)


def _plot_measurements_by_score(ax, samples, score_key, title, cmap='viridis'):
    if not samples:
        ax.set_title(title + '\nnone')
        return
    segs = [((d['x1'], d['y1']), (d['x2'], d['y2'])) for d in samples]
    values = np.asarray([d.get(score_key, 0.0) for d in samples], float)
    lc = LineCollection(segs, cmap=cmap, linewidths=0.9, alpha=0.9)
    lc.set_array(values)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
    ax.set_title(f'{title}\nn={len(samples)}')


def _draw_region_overlay(ax, regions, representatives=None, labels=False):
    cmap = plt.get_cmap('tab20')
    for idx, region in enumerate(regions):
        seq = region['samples']
        color = cmap((region['fiber_region_id'] - 1) % 20)
        centers = np.array([[d['center_x'], d['center_y']] for d in seq], float)
        e1 = np.array([[d['x1'], d['y1']] for d in seq], float)
        e2 = np.array([[d['x2'], d['y2']] for d in seq], float)
        if len(seq) >= 2:
            ax.plot(centers[:, 0], centers[:, 1], color=color, linewidth=1.25, alpha=0.95)
            ax.plot(e1[:, 0], e1[:, 1], color=color, linewidth=0.55, alpha=0.65)
            ax.plot(e2[:, 0], e2[:, 1], color=color, linewidth=0.55, alpha=0.65)
        ax.scatter([centers[0, 0], centers[-1, 0]], [centers[0, 1], centers[-1, 1]],
                   s=10, color=[color], marker='o')
        if labels and idx < REGION_LABEL_LIMIT:
            mid = centers[len(centers) // 2]
            ax.text(mid[0], mid[1], str(region['fiber_region_id']), color='white',
                    fontsize=5, ha='center', va='center',
                    bbox=dict(facecolor=color, edgecolor='none', alpha=0.65, pad=0.7))
    if representatives is not None and len(representatives):
        add_chords(ax, representatives.to_dict('records'), color='white', linewidth=1.8,
                   label='representative width')


def draw_fiber_region_result(img, ridge_response, ridge_centerline,
                             small_ridge_response, small_ridge_centerline,
                             edge_samples, ridge_samples, bundle_rejected,
                             sem_refined, sem_downgraded, regions,
                             region_df, representative_df, name, output_png):
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    for ax in axes.ravel():
        ax.imshow(img, cmap='gray')
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis('off')

    axes[0, 0].imshow(ridge_response, cmap='magma', alpha=0.40)
    axes[0, 0].imshow(np.ma.masked_where(~ridge_centerline, ridge_centerline),
                      cmap='winter', alpha=0.95)
    axes[0, 0].set_title(f'1. broad centreline hypotheses\npixels={int(ridge_centerline.sum())}')

    axes[0, 1].imshow(small_ridge_response, cmap='magma', alpha=0.27)
    axes[0, 1].imshow(np.ma.masked_where(~small_ridge_centerline, small_ridge_centerline),
                      cmap='winter', alpha=0.95)
    axes[0, 1].set_title(f'2. small-scale centreline evidence\npixels={int(small_ridge_centerline.sum())}')

    add_chords(axes[0, 2], edge_samples, color='cyan', linewidth=0.55, label='edge')
    add_chords(axes[0, 2], ridge_samples, color='magenta', linewidth=0.55, label='ridge')
    axes[0, 2].legend(fontsize=7, loc='upper right')
    axes[0, 2].set_title(f'3. raw local detections\nedge={len(edge_samples)}, ridge={len(ridge_samples)}')

    _plot_measurements_by_score(axes[1, 0], sem_refined, 'sem_agreement',
                                '4. SEM-refined local thickness')

    _draw_region_overlay(axes[1, 1], regions, labels=True)
    axes[1, 1].set_title(f'5. continuous fibre regions\nregions={len(regions)}')

    _draw_region_overlay(axes[1, 2], regions, representative_df, labels=False)
    axes[1, 2].set_title(f'6. one or more representatives per region\nrepresentatives={len(representative_df)}')

    show_bad = sorted(sem_downgraded, key=lambda d: d.get('confidence', 0), reverse=True)[:REGION_CANDIDATE_DRAW_LIMIT]
    add_chords(axes[2, 0], show_bad, color='orange', linewidth=0.38, label='SEM/bundle candidate')
    show_bundle = sorted(bundle_rejected, key=lambda d: d.get('bundle_score', 0), reverse=True)[:350]
    add_chords(axes[2, 0], show_bundle, color='red', linewidth=0.35, label='bundle rejected')
    if show_bad or show_bundle:
        axes[2, 0].legend(fontsize=7, loc='upper right')
    axes[2, 0].set_title(f'7. excluded/ambiguous local candidates\nshown={len(show_bad)+len(show_bundle)}')

    axes[2, 1].clear(); axes[2, 1].axis('on'); axes[2, 1].set_aspect('auto')
    local_widths = [d['width_px'] for d in sem_refined]
    if local_widths:
        axes[2, 1].hist(local_widths, bins='auto', alpha=0.72)
    axes[2, 1].set_xlabel('local thickness (px)')
    axes[2, 1].set_ylabel('local measurement count')
    axes[2, 1].set_title('8. diagnostic local-thickness distribution\n(not used as fibre-count distribution)')

    axes[2, 2].clear(); axes[2, 2].axis('on'); axes[2, 2].set_aspect('auto')
    if len(representative_df):
        vals = representative_df['representative_width_px'].to_numpy(float)
        weights = representative_df['fiber_count_weight'].to_numpy(float)
        axes[2, 2].hist(vals, bins=weighted_histogram_bin_count(vals), weights=weights, alpha=0.72)
        med = _weighted_median(vals, weights)
        axes[2, 2].axvline(med, linestyle='--', linewidth=1.2,
                           label=f'weighted median={med:.2f}px')
        axes[2, 2].legend(fontsize=8)
    axes[2, 2].set_xlabel('representative thickness (px)')
    axes[2, 2].set_ylabel('fibre-region count weight')
    axes[2, 2].set_title('9. fibre-region representative distribution\n(each region has total weight 1)')

    fig.suptitle(f'{name} | SEM-refined fibre regions and representative widths', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches='tight')
    if SHOW_INLINE:
        plt.show()
    plt.close(fig)


def process_one(path):
    img, cut = prepare_image(path)
    gy, gx, grad, theta, coh, energy = orientation_fields(img)
    continuous_edge = build_continuous_edges(img, coh, energy)
    dense_edge, dense_threshold = build_dense_edge_candidates(grad, gy, gx, coh, energy)
    edge_curves, edge_junctions = trace_edge_curves(continuous_edge)

    thin_samples, thin_rejected, thin_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'thin')
    thick_samples, thick_rejected, thick_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'thick')
    curved_samples, curved_rejected, curved_max = measure_branch(
        img, gy, gx, grad, theta, coh, continuous_edge, dense_edge,
        edge_curves, edge_junctions, 'curved')
    edge_samples = [canonicalize_measurement(v, 'edge') for v in
                    merge_method_results(thin_samples, thick_samples, curved_samples)]
    thin_lock_model = build_thin_lock_model(edge_samples)

    (ridge_response, ridge_centerline, ridge_curves, ridge_junctions,
     small_ridge_response, small_ridge_centerline, small_ridge_curves,
     small_ridge_junctions) = build_bright_ridge_centerlines(img)
    small_ridge_dist = ndimage.distance_transform_edt(~small_ridge_centerline)

    ridge_samples, ridge_rejected, ridge_max, bundle_rejected = measure_ridge_detector(
        img, ridge_response, ridge_curves, gy, gx, grad, theta, coh,
        continuous_edge, dense_edge, small_ridge_dist, thin_lock_model)

    confirmed, ridge_only, edge_only = match_detector_consensus(ridge_samples, edge_samples)
    anchors = build_tracking_anchors(confirmed, ridge_only, edge_only)
    block_b, block_c, track_meta = run_adaptive_block_tracks(
        img, ridge_response, theta, coh, gy, gx, grad,
        continuous_edge, dense_edge, edge_junctions,
        small_ridge_dist, thin_lock_model, anchors,
        max(thin_max, thick_max, curved_max, ridge_max))
    local_final, initial_candidates = merge_final_measurements(
        confirmed, block_b, ridge_only + edge_only + block_c)

    sem_refined, sem_downgraded = refine_measurements_against_sem(
        img, ridge_response, theta, coh, gy, gx, grad,
        continuous_edge, dense_edge, small_ridge_dist,
        thin_lock_model, local_final)
    regions = build_fiber_regions(sem_refined, ridge_curves, small_ridge_curves)
    nm_px = NM_PER_PX.get(path.name)
    region_df, representative_df, local_df = summarize_fiber_regions(regions, nm_px)

    if len(local_df):
        local_df.insert(0, 'file', path.name)
        local_df['status'] = 'region_local_measurement'
        local_df['nm_per_px'] = nm_px if nm_px is not None else np.nan
        local_df['thickness_nm'] = local_df['width_px'] * nm_px if nm_px is not None else np.nan
    if len(region_df):
        region_df.insert(0, 'file', path.name)
    if len(representative_df):
        representative_df.insert(0, 'file', path.name)
        representative_df['nm_per_px'] = nm_px if nm_px is not None else np.nan
    candidate_df = pd.DataFrame(initial_candidates + sem_downgraded)
    if len(candidate_df):
        candidate_df.insert(0, 'file', path.name)
        candidate_df['status'] = 'candidate_or_sem_downgraded'
    detector_df = pd.DataFrame(edge_samples + ridge_samples)
    if len(detector_df):
        detector_df.insert(0, 'file', path.name)
    rejection_df = pd.DataFrame(thin_rejected + thick_rejected + curved_rejected + ridge_rejected)
    if len(rejection_df):
        rejection_df.insert(0, 'file', path.name)
    track_df = pd.DataFrame(track_meta)
    if len(track_df):
        track_df.insert(0, 'file', path.name)
    bundle_df = pd.DataFrame(bundle_rejected)
    if len(bundle_df):
        bundle_df.insert(0, 'file', path.name)
        bundle_df['status'] = 'bundle_rejected'

    output_png = OUTPUT_DIR / f'{path.stem}_fiber_regions_sem_refined.png'
    draw_fiber_region_result(
        img, ridge_response, ridge_centerline,
        small_ridge_response, small_ridge_centerline,
        edge_samples, ridge_samples, bundle_rejected,
        sem_refined, sem_downgraded, regions,
        region_df, representative_df, path.name, output_png)

    local_df.to_csv(OUTPUT_DIR / f'{path.stem}_region_local_measurements.csv', index=False)
    region_df.to_csv(OUTPUT_DIR / f'{path.stem}_fiber_regions.csv', index=False)
    representative_df.to_csv(OUTPUT_DIR / f'{path.stem}_fiber_region_representatives.csv', index=False)
    candidate_df.to_csv(OUTPUT_DIR / f'{path.stem}_excluded_candidates.csv', index=False)
    detector_df.to_csv(OUTPUT_DIR / f'{path.stem}_raw_detector_measurements.csv', index=False)
    track_df.to_csv(OUTPUT_DIR / f'{path.stem}_block_track_summary.csv', index=False)
    rejection_df.to_csv(OUTPUT_DIR / f'{path.stem}_rejected_runs.csv', index=False)
    bundle_df.to_csv(OUTPUT_DIR / f'{path.stem}_bundle_rejected_measurements.csv', index=False)

    if len(representative_df):
        rep_vals = representative_df['representative_width_px'].to_numpy(float)
        rep_weights = representative_df['fiber_count_weight'].to_numpy(float)
        representative_median = _weighted_median(rep_vals, rep_weights)
    else:
        representative_median = np.nan
    summary = dict(
        file=path.name, cropped_height=cut,
        image_height=img.shape[0], image_width=img.shape[1],
        edge_detector_measurements=len(edge_samples),
        ridge_detector_measurements=len(ridge_samples),
        initial_local_A_B_measurements=len(local_final),
        sem_refined_local_measurements=len(sem_refined),
        sem_downgraded_measurements=len(sem_downgraded),
        fiber_regions=len(region_df),
        representative_widths=len(representative_df),
        representative_weighted_median_px=representative_median,
        bundle_rejected_measurements=len(bundle_df),
        block_anchors=len(anchors),
        dense_gradient_threshold=dense_threshold,
    )
    return summary, local_df, region_df, representative_df, candidate_df


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f'Folder not found: {IMAGE_DIR}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files, missing = [], []
    for name in TARGET_FILES:
        p = IMAGE_DIR / name
        (files if p.is_file() else missing).append(p if p.is_file() else name)
    if missing:
        raise FileNotFoundError('Requested image(s) not found: ' + ', '.join(map(str, missing)))
    print(f'Input : {IMAGE_DIR}\nOutput: {OUTPUT_DIR}')
    print('Images: ' + ', '.join(p.name for p in files))
    summaries, local_frames, region_frames, representative_frames, candidate_frames = [], [], [], [], []
    for i, path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] {path.name}')
        try:
            summary, local_df, region_df, rep_df, candidate_df = process_one(path)
            summaries.append(summary)
            if len(local_df): local_frames.append(local_df)
            if len(region_df): region_frames.append(region_df)
            if len(rep_df): representative_frames.append(rep_df)
            if len(candidate_df): candidate_frames.append(candidate_df)
            print(f"    local={summary['sem_refined_local_measurements']}, "
                  f"regions={summary['fiber_regions']}, "
                  f"representatives={summary['representative_widths']}, "
                  f"weighted median={summary['representative_weighted_median_px']:.2f}px")
        except Exception as exc:
            warnings.warn(f'{path.name}: {type(exc).__name__}: {exc}')
            summaries.append(dict(file=path.name, error=f'{type(exc).__name__}: {exc}'))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / 'fiber_region_summary.csv', index=False)
    if local_frames:
        pd.concat(local_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_region_local_measurements.csv', index=False)
    if region_frames:
        pd.concat(region_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_fiber_regions.csv', index=False)
    if representative_frames:
        all_reps = pd.concat(representative_frames, ignore_index=True)
        all_reps.to_csv(OUTPUT_DIR / 'all_fiber_region_representatives.csv', index=False)
        # A compact distribution table directly usable by later 3-D reconstruction.
        all_reps[['file', 'fiber_region_id', 'subregion_id',
                  'representative_width_px', 'fiber_count_weight',
                  'region_length_fraction', 'length_weight'] +
                 ([ 'representative_width_nm' ] if 'representative_width_nm' in all_reps.columns else [])].to_csv(
            OUTPUT_DIR / 'fiber_thickness_distribution_for_3d.csv', index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / 'all_excluded_candidates.csv', index=False)
    print('\nDone. Results saved under:')
    print(OUTPUT_DIR)
    display(summary_df) if 'display' in globals() else print(summary_df)


if __name__ == '__main__':
    main()
