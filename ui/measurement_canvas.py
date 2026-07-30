from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

_COMPONENT = None


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    return np.round(np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0, 1) * 255).astype(np.uint8)


def image_to_data_url(image: np.ndarray) -> str:
    arr = _as_uint8(image)
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr if arr.ndim == 2 else arr[..., :3], mode=mode)
    buffer = BytesIO()
    pil.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_canvas_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"new_measurements": [], "delete_ids": []}
    new_items = []
    for item in payload.get("new_measurements", []) or []:
        if not isinstance(item, dict):
            continue
        p1, p2 = item.get("p1"), item.get("p2")
        if not (
            isinstance(p1, (list, tuple))
            and isinstance(p2, (list, tuple))
            and len(p1) == 2
            and len(p2) == 2
        ):
            continue
        try:
            new_items.append({
                "p1": [float(p1[0]), float(p1[1])],
                "p2": [float(p2[0]), float(p2[1])],
            })
        except (TypeError, ValueError):
            continue
    delete_ids = []
    seen = set()
    for value in payload.get("delete_ids", []) or []:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            delete_ids.append(text)
    return {"new_measurements": new_items, "delete_ids": delete_ids}


_HTML = r"""
<div class="vf-shell">
  <div class="vf-toolbar" role="toolbar" aria-label="측정 도구">
    <button data-tool="pan" title="이동: 이미지를 끌어서 이동합니다. 마우스 휠로 확대·축소할 수 있습니다.">↔ 이동 <span class="vf-q">?</span></button>
    <button data-tool="add" class="active" title="두께 추가: fiber의 한쪽 edge와 반대쪽 edge를 차례로 클릭합니다.">＋ 두께 추가 <span class="vf-q">?</span></button>
    <button data-tool="erase" title="지우개: 삭제할 자동 또는 수동 두께선을 클릭합니다.">⌫ 지우개 <span class="vf-q">?</span></button>
    <span class="vf-divider"></span>
    <button data-action="undo" title="실행 취소: 마지막 추가 또는 삭제 작업을 되돌립니다.">↶ 실행 취소 <span class="vf-q">?</span></button>
    <button data-action="clear" title="임시 측정 초기화: 아직 반영하지 않은 수동 두께선만 모두 지웁니다.">임시선 지우기 <span class="vf-q">?</span></button>
    <button data-action="fit" title="화면 맞춤: SEM 이미지 전체가 보이도록 확대 배율을 초기화합니다.">화면 맞춤 <span class="vf-q">?</span></button>
    <span class="vf-spacer"></span>
    <button data-action="apply" class="primary" title="전체 반영: 추가한 수동 측정과 지운 측정을 한 번에 두께 분포에 적용합니다.">전체 반영 <span class="vf-q">?</span></button>
  </div>
  <div class="vf-stage" tabindex="0">
    <canvas></canvas>
    <div class="vf-hint"></div>
  </div>
  <div class="vf-status">
    <span class="vf-mode"></span>
    <span class="vf-counts"></span>
  </div>
</div>
"""

_CSS = r"""
.vf-shell { width:100%; font-family:var(--st-font, sans-serif); color:var(--st-text-color); }
.vf-toolbar { display:flex; gap:.42rem; align-items:center; flex-wrap:wrap; padding:.55rem .6rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:.75rem .75rem 0 0; background:color-mix(in srgb, var(--st-secondary-background-color) 92%, transparent); }
.vf-toolbar button { border:1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent); background:var(--st-background-color); color:var(--st-text-color); border-radius:.48rem; padding:.42rem .7rem; cursor:pointer; font-size:.88rem; }
.vf-toolbar button:hover { border-color:var(--st-primary-color); }
.vf-toolbar button.active { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); }
.vf-toolbar button.primary { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); font-weight:650; }
.vf-toolbar button:disabled { opacity:.45; cursor:not-allowed; }
.vf-q { display:inline-grid; place-items:center; width:1rem; height:1rem; margin-left:.2rem; border-radius:50%; border:1px solid currentColor; font-size:.68rem; opacity:.72; }
.vf-divider { width:1px; height:1.65rem; background:color-mix(in srgb, var(--st-text-color) 18%, transparent); }
.vf-spacer { flex:1; }
.vf-stage { position:relative; width:100%; height:min(72vh, 780px); min-height:480px; overflow:hidden; background:#0b0d10; border-left:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-right:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); outline:none; }
.vf-stage canvas { display:block; width:100%; height:100%; touch-action:none; }
.vf-hint { position:absolute; left:50%; top:.7rem; transform:translateX(-50%); background:rgba(0,0,0,.62); color:white; padding:.34rem .62rem; border-radius:999px; font-size:.78rem; pointer-events:none; opacity:.9; }
.vf-status { display:flex; justify-content:space-between; gap:1rem; padding:.45rem .65rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:0 0 .75rem .75rem; color:color-mix(in srgb, var(--st-text-color) 78%, transparent); font-size:.8rem; }
@media (max-width: 760px) { .vf-stage { min-height:380px; height:62vh; } .vf-toolbar button { padding:.4rem .55rem; } }
"""

_JS = r"""
export default function(component) {
  const { parentElement, data, setTriggerValue } = component;
  const shell = parentElement.querySelector('.vf-shell');
  const stage = shell.querySelector('.vf-stage');
  const canvas = shell.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const hint = shell.querySelector('.vf-hint');
  const modeLabel = shell.querySelector('.vf-mode');
  const countsLabel = shell.querySelector('.vf-counts');
  const applyButton = shell.querySelector('[data-action="apply"]');

  if (shell.__vfCleanup) shell.__vfCleanup();

  const image = new Image();
  const committed = Array.isArray(data.lines) ? data.lines : [];
  let mode = 'add';
  let pending = [];
  let deleted = new Set();
  let firstPoint = null;
  let history = [];
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let dragging = false;
  let dragStart = null;
  let dragOrigin = null;
  let destroyed = false;
  let dpr = Math.max(1, window.devicePixelRatio || 1);

  const analysisScale = Number(data.analysis_scale || 1);
  const nmPerPx = data.nm_per_px == null ? null : Number(data.nm_per_px);

  function snapshot() {
    history.push({
      pending: JSON.parse(JSON.stringify(pending)),
      deleted: Array.from(deleted),
      firstPoint: firstPoint ? {...firstPoint} : null,
    });
    if (history.length > 50) history.shift();
  }

  function restoreLast() {
    const previous = history.pop();
    if (!previous) return;
    pending = previous.pending;
    deleted = new Set(previous.deleted);
    firstPoint = previous.firstPoint;
    draw();
  }

  function resizeCanvas() {
    const rect = stage.getBoundingClientRect();
    dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    draw();
  }

  function fitImage() {
    const rect = stage.getBoundingClientRect();
    if (!image.naturalWidth || !rect.width) return;
    scale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight);
    scale = Math.max(scale, 0.02);
    offsetX = (rect.width - image.naturalWidth * scale) / 2;
    offsetY = (rect.height - image.naturalHeight * scale) / 2;
    draw();
  }

  function imageToScreen(point) {
    return {x: point.x * scale + offsetX, y: point.y * scale + offsetY};
  }

  function screenToImage(x, y) {
    return {
      x: Math.max(0, Math.min(image.naturalWidth - 1, (x - offsetX) / scale)),
      y: Math.max(0, Math.min(image.naturalHeight - 1, (y - offsetY) / scale)),
    };
  }

  function lineDistance(px, py, line) {
    const a = imageToScreen({x:Number(line.x1 ?? line.p1?.[0]), y:Number(line.y1 ?? line.p1?.[1])});
    const b = imageToScreen({x:Number(line.x2 ?? line.p2?.[0]), y:Number(line.y2 ?? line.p2?.[1])});
    const vx = b.x - a.x, vy = b.y - a.y;
    const wx = px - a.x, wy = py - a.y;
    const denom = vx * vx + vy * vy;
    const t = denom > 0 ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / denom)) : 0;
    return Math.hypot(px - (a.x + t * vx), py - (a.y + t * vy));
  }

  function widthText(line) {
    const p1 = line.p1 || [line.x1, line.y1];
    const p2 = line.p2 || [line.x2, line.y2];
    const analysisWidth = Math.hypot(Number(p2[0]) - Number(p1[0]), Number(p2[1]) - Number(p1[1]));
    const originalWidth = line.width_original_px != null ? Number(line.width_original_px) : analysisWidth / analysisScale;
    if (nmPerPx != null && Number.isFinite(nmPerPx)) return `${(originalWidth * nmPerPx).toFixed(2)} nm`;
    return `${originalWidth.toFixed(2)} px`;
  }

  function drawLine(line, color, lineWidth, dashed=false, label=false) {
    const p1 = line.p1 || [line.x1, line.y1];
    const p2 = line.p2 || [line.x2, line.y2];
    const a = imageToScreen({x:Number(p1[0]), y:Number(p1[1])});
    const b = imageToScreen({x:Number(p2[0]), y:Number(p2[1])});
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dashed ? [7, 5] : []);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.fillStyle = color;
    for (const p of [a,b]) { ctx.beginPath(); ctx.arc(p.x,p.y,Math.max(2.4,lineWidth+1),0,Math.PI*2); ctx.fill(); }
    if (label) {
      const text = widthText(line);
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      ctx.font = '12px sans-serif';
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = 'rgba(0,0,0,.72)'; ctx.fillRect(mx - tw/2 - 4, my - 20, tw + 8, 17);
      ctx.fillStyle = '#fff'; ctx.fillText(text, mx - tw/2, my - 7);
    }
    ctx.restore();
  }

  function draw() {
    if (destroyed || !canvas.width) return;
    const rect = stage.getBoundingClientRect();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = '#0b0d10'; ctx.fillRect(0,0,rect.width,rect.height);
    if (image.complete && image.naturalWidth) {
      ctx.imageSmoothingEnabled = scale < 1;
      ctx.drawImage(image, offsetX, offsetY, image.naturalWidth * scale, image.naturalHeight * scale);
    }

    for (const line of committed) {
      const eraseIds = Array.isArray(line.erase_ids) ? line.erase_ids.map(String) : [];
      const isDeleted = eraseIds.some(id => deleted.has(id));
      let color = line.source === 'manual' ? '#16d9e8' : (line.source === 'orientation' ? '#a6ecff' : '#ffd54a');
      if (isDeleted) color = 'rgba(255,72,86,.9)';
      drawLine(line, color, line.source === 'manual' ? 2.8 : 2.1, isDeleted, false);
    }
    pending.forEach((line, index) => drawLine(line, '#44a3ff', 2.8, true, true));
    if (firstPoint) {
      const p = imageToScreen(firstPoint);
      ctx.fillStyle = '#44a3ff'; ctx.beginPath(); ctx.arc(p.x,p.y,5,0,Math.PI*2); ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.2; ctx.stroke();
    }
    updateStatus();
  }

  function updateStatus() {
    const labels = {
      pan: '이동 모드 · 드래그로 이동, 휠로 확대',
      add: firstPoint ? '두 번째 edge를 클릭하세요' : 'fiber의 첫 번째 edge를 클릭하세요',
      erase: '지울 두께선을 클릭하세요',
    };
    modeLabel.textContent = labels[mode];
    hint.textContent = labels[mode];
    countsLabel.textContent = `임시 측정 ${pending.length}개 · 삭제 예정 ${deleted.size}개`;
    applyButton.disabled = pending.length === 0 && deleted.size === 0;
  }

  function setMode(nextMode) {
    mode = nextMode;
    firstPoint = null;
    shell.querySelectorAll('[data-tool]').forEach(btn => btn.classList.toggle('active', btn.dataset.tool === mode));
    stage.style.cursor = mode === 'pan' ? 'grab' : (mode === 'erase' ? 'not-allowed' : 'crosshair');
    draw();
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {x:event.clientX - rect.left, y:event.clientY - rect.top};
  }

  function onPointerDown(event) {
    stage.focus();
    const p = pointerPosition(event);
    if (mode === 'pan' || event.button === 1 || event.button === 2) {
      dragging = true; dragStart = p; dragOrigin = {x:offsetX,y:offsetY};
      stage.setPointerCapture(event.pointerId); stage.style.cursor='grabbing';
      return;
    }
    if (event.button !== 0) return;
    if (mode === 'add') {
      const point = screenToImage(p.x,p.y);
      if (!firstPoint) {
        firstPoint = point;
      } else {
        const distance = Math.hypot(point.x-firstPoint.x, point.y-firstPoint.y);
        if (distance >= 0.75) {
          snapshot();
          pending.push({p1:[firstPoint.x,firstPoint.y], p2:[point.x,point.y]});
        }
        firstPoint = null;
      }
      draw();
      return;
    }
    if (mode === 'erase') {
      let best = null;
      pending.forEach((line,index) => {
        const d = lineDistance(p.x,p.y,line);
        if (!best || d < best.distance) best={kind:'pending',index,distance:d};
      });
      committed.forEach((line,index) => {
        const d = lineDistance(p.x,p.y,line);
        if (!best || d < best.distance) best={kind:'committed',index,distance:d};
      });
      if (best && best.distance <= 14) {
        snapshot();
        if (best.kind === 'pending') pending.splice(best.index,1);
        else {
          const ids = Array.isArray(committed[best.index].erase_ids) ? committed[best.index].erase_ids.map(String) : [];
          const already = ids.length && ids.every(id => deleted.has(id));
          ids.forEach(id => already ? deleted.delete(id) : deleted.add(id));
        }
        draw();
      }
    }
  }

  function onPointerMove(event) {
    if (!dragging) return;
    const p = pointerPosition(event);
    offsetX = dragOrigin.x + (p.x - dragStart.x);
    offsetY = dragOrigin.y + (p.y - dragStart.y);
    draw();
  }

  function onPointerUp(event) {
    if (!dragging) return;
    dragging = false;
    try { stage.releasePointerCapture(event.pointerId); } catch (_) {}
    stage.style.cursor = mode === 'pan' ? 'grab' : (mode === 'erase' ? 'not-allowed' : 'crosshair');
  }

  function onWheel(event) {
    event.preventDefault();
    const p = pointerPosition(event);
    const before = screenToImage(p.x,p.y);
    const factor = Math.exp(-event.deltaY * 0.0015);
    scale = Math.max(0.05, Math.min(24, scale * factor));
    offsetX = p.x - before.x * scale;
    offsetY = p.y - before.y * scale;
    draw();
  }

  function onKeyDown(event) {
    if (event.key === 'Escape') { firstPoint = null; draw(); }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); restoreLast(); }
    if (event.key === '1') setMode('pan');
    if (event.key === '2') setMode('add');
    if (event.key === '3') setMode('erase');
  }

  shell.querySelectorAll('[data-tool]').forEach(button => {
    button.onclick = () => setMode(button.dataset.tool);
  });
  shell.querySelector('[data-action="undo"]').onclick = restoreLast;
  shell.querySelector('[data-action="clear"]').onclick = () => {
    if (!pending.length && !firstPoint) return;
    snapshot(); pending=[]; firstPoint=null; draw();
  };
  shell.querySelector('[data-action="fit"]').onclick = fitImage;
  applyButton.onclick = () => {
    if (!pending.length && !deleted.size) return;
    applyButton.disabled = true;
    setTriggerValue('apply', {
      new_measurements: pending.map(line => ({p1:line.p1,p2:line.p2})),
      delete_ids: Array.from(deleted),
    });
  };

  stage.addEventListener('pointerdown', onPointerDown);
  stage.addEventListener('pointermove', onPointerMove);
  stage.addEventListener('pointerup', onPointerUp);
  stage.addEventListener('pointercancel', onPointerUp);
  stage.addEventListener('wheel', onWheel, {passive:false});
  stage.addEventListener('keydown', onKeyDown);
  stage.addEventListener('contextmenu', event => event.preventDefault());
  const observer = new ResizeObserver(resizeCanvas);
  observer.observe(stage);
  image.onload = () => { resizeCanvas(); fitImage(); };
  image.src = String(data.image_url || '');
  setMode('add');

  const cleanup = () => {
    destroyed = true;
    observer.disconnect();
    stage.removeEventListener('pointerdown', onPointerDown);
    stage.removeEventListener('pointermove', onPointerMove);
    stage.removeEventListener('pointerup', onPointerUp);
    stage.removeEventListener('pointercancel', onPointerUp);
    stage.removeEventListener('wheel', onWheel);
    stage.removeEventListener('keydown', onKeyDown);
  };
  shell.__vfCleanup = cleanup;
  return cleanup;
}
"""


def measurement_canvas(
    image: np.ndarray,
    lines: list[dict],
    analysis_scale: float,
    nm_per_px: float | None,
    revision: int,
    *,
    key: str,
):
    """Render the local-interaction canvas. Python reruns only when '전체 반영' is pressed."""
    import streamlit as st

    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = st.components.v2.component(
            "visionflux.measurement_canvas",
            html=_HTML,
            css=_CSS,
            js=_JS,
            isolate_styles=True,
        )
    return _COMPONENT(
        data={
            "image_url": image_to_data_url(image),
            "lines": lines,
            "analysis_scale": float(analysis_scale),
            "nm_per_px": None if nm_per_px is None else float(nm_per_px),
            "revision": int(revision),
        },
        key=key,
        on_apply_change=lambda: None,
    )
