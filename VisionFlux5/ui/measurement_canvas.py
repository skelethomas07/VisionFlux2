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
    <button data-tool="pan" title="이동/선택: 드래그로 이동하고, 두께선을 클릭하면 해당 fiber를 강조합니다.">↔ 이동·선택 <span class="vf-q">?</span></button>
    <button data-tool="add" class="active" title="두께 추가: fiber의 양쪽 edge를 차례로 클릭합니다.">＋ 두께 추가 <span class="vf-q">?</span></button>
    <button data-tool="erase" title="지우개: 삭제할 자동 또는 수동 두께선을 클릭합니다.">⌫ 지우개 <span class="vf-q">?</span></button>
    <span class="vf-divider"></span>
    <button data-action="undo" title="마지막 추가 또는 삭제 작업을 되돌립니다.">↶ 실행 취소</button>
    <button data-action="clear" title="아직 반영하지 않은 수동 두께선만 지웁니다.">임시선 지우기</button>
    <button data-action="fit" title="현재 선택한 섹터 또는 전체 이미지에 화면을 맞춥니다.">화면 맞춤</button>
    <button data-action="magnifier" class="active" title="커서 주변을 작은 창으로 확대합니다.">⌕ 돋보기 ON</button>
    <span class="vf-divider"></span>
    <label class="vf-sector-label" title="전체 이미지 또는 6개 이상의 섹터로 나누어 검토합니다.">검토 보기
      <select data-action="sector-layout">
        <option value="1x1">전체</option>
        <option value="3x2">6개</option>
        <option value="3x3">9개</option>
        <option value="4x3">12개</option>
        <option value="4x4">16개</option>
      </select>
    </label>
    <button data-action="prev-sector" title="이전 섹터로 이동합니다.">이전</button>
    <button data-action="next-sector" title="현재 섹터를 완료 처리하고 다음 섹터로 이동합니다.">완료·다음</button>
    <span class="vf-sector-progress"></span>
    <span class="vf-spacer"></span>
    <button data-action="apply" class="primary" title="추가·삭제한 측정값을 한 번에 최종 결과에 반영합니다.">전체 반영 <span class="vf-q">?</span></button>
  </div>
  <div class="vf-stage" tabindex="0">
    <canvas class="vf-main-canvas"></canvas>
    <canvas class="vf-magnifier" width="190" height="190"></canvas>
    <div class="vf-selection"></div>
    <div class="vf-hint"></div>
  </div>
  <div class="vf-status">
    <span class="vf-mode"></span>
    <span class="vf-counts"></span>
    <span class="vf-save-status"></span>
  </div>
</div>
"""

_CSS = r"""
.vf-shell { width:100%; font-family:var(--st-font, sans-serif); color:var(--st-text-color); }
.vf-toolbar { display:flex; gap:.38rem; align-items:center; flex-wrap:wrap; padding:.55rem .6rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:.75rem .75rem 0 0; background:color-mix(in srgb, var(--st-secondary-background-color) 92%, transparent); }
.vf-toolbar button, .vf-toolbar select { border:1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent); background:var(--st-background-color); color:var(--st-text-color); border-radius:.48rem; padding:.40rem .62rem; cursor:pointer; font-size:.84rem; }
.vf-toolbar button:hover { border-color:var(--st-primary-color); }
.vf-toolbar button.active { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); }
.vf-toolbar button.primary { color:white; background:var(--st-primary-color); border-color:var(--st-primary-color); font-weight:650; }
.vf-toolbar button:disabled { opacity:.45; cursor:not-allowed; }
.vf-q { display:inline-grid; place-items:center; width:1rem; height:1rem; margin-left:.18rem; border-radius:50%; border:1px solid currentColor; font-size:.68rem; opacity:.72; }
.vf-divider { width:1px; height:1.65rem; background:color-mix(in srgb, var(--st-text-color) 18%, transparent); }
.vf-spacer { flex:1; }
.vf-sector-label { display:flex; gap:.35rem; align-items:center; font-size:.82rem; }
.vf-sector-progress { font-size:.78rem; min-width:5.5rem; }
.vf-stage { position:relative; width:100%; height:min(74vh, 800px); min-height:500px; overflow:hidden; background:#0b0d10; border-left:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-right:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); outline:none; }
.vf-main-canvas { display:block; width:100%; height:100%; touch-action:none; }
.vf-magnifier { position:absolute; right:14px; top:14px; width:190px; height:190px; border:2px solid white; border-radius:10px; background:black; box-shadow:0 4px 18px rgba(0,0,0,.45); pointer-events:none; display:none; }
.vf-selection { position:absolute; left:12px; bottom:12px; max-width:330px; background:rgba(0,0,0,.72); color:white; padding:.55rem .7rem; border-radius:.6rem; font-size:.8rem; pointer-events:none; display:none; }
.vf-hint { position:absolute; left:50%; top:.7rem; transform:translateX(-50%); background:rgba(0,0,0,.62); color:white; padding:.34rem .62rem; border-radius:999px; font-size:.78rem; pointer-events:none; opacity:.9; }
.vf-status { display:flex; justify-content:space-between; gap:1rem; padding:.45rem .65rem; border:1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent); border-radius:0 0 .75rem .75rem; color:color-mix(in srgb, var(--st-text-color) 78%, transparent); font-size:.8rem; }
@media (max-width: 760px) { .vf-stage { min-height:390px; height:64vh; } .vf-toolbar button { padding:.38rem .5rem; } .vf-magnifier { width:145px; height:145px; } }
"""

_JS = r"""
export default function(component) {
  const { parentElement, data, setTriggerValue } = component;
  const shell = parentElement.querySelector('.vf-shell');
  const stage = shell.querySelector('.vf-stage');
  const canvas = shell.querySelector('.vf-main-canvas');
  const ctx = canvas.getContext('2d');
  const magnifier = shell.querySelector('.vf-magnifier');
  const mctx = magnifier.getContext('2d');
  const hint = shell.querySelector('.vf-hint');
  const selectionPanel = shell.querySelector('.vf-selection');
  const modeLabel = shell.querySelector('.vf-mode');
  const countsLabel = shell.querySelector('.vf-counts');
  const saveLabel = shell.querySelector('.vf-save-status');
  const sectorProgress = shell.querySelector('.vf-sector-progress');
  const sectorSelect = shell.querySelector('[data-action="sector-layout"]');
  const applyButton = shell.querySelector('[data-action="apply"]');
  const magnifierButton = shell.querySelector('[data-action="magnifier"]');

  if (shell.__vfCleanup) shell.__vfCleanup();

  const image = new Image();
  const committed = Array.isArray(data.lines) ? data.lines : [];
  const analysisScale = Number(data.analysis_scale || 1);
  const nmPerPx = data.nm_per_px == null ? null : Number(data.nm_per_px);
  const revision = Number(data.revision || 0);
  const storageKey = `visionflux:${String(data.autosave_key || 'default')}`;

  let mode = 'add';
  let pending = [];
  let deleted = new Set();
  let firstPoint = null;
  let history = [];
  let scale = 1, offsetX = 0, offsetY = 0;
  let dragging = false, dragStart = null, dragOrigin = null, moved = false;
  let dpr = Math.max(1, window.devicePixelRatio || 1);
  let selectedId = null;
  let pointerImage = null;
  let magnifierEnabled = true;
  let sectorLayout = '1x1';
  let sectorIndex = 0;
  let completedSectors = new Set();

  function snapshot() {
    history.push({pending:structuredClone(pending), deleted:Array.from(deleted), firstPoint:firstPoint ? {...firstPoint}:null});
    if (history.length > 60) history.shift();
  }
  function restoreLast() {
    const previous = history.pop(); if (!previous) return;
    pending = previous.pending; deleted = new Set(previous.deleted); firstPoint = previous.firstPoint; draw(); queueSave();
  }
  function saveState() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({revision,pending,deleted:Array.from(deleted),sectorLayout,sectorIndex,completedSectors:Array.from(completedSectors),magnifierEnabled,savedAt:Date.now()}));
      const now = new Date(); saveLabel.textContent = `임시저장 ${now.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;
    } catch (_) { saveLabel.textContent = '임시저장 불가'; }
  }
  let saveTimer = null;
  function queueSave() { clearTimeout(saveTimer); saveTimer = setTimeout(saveState, 1200); }
  function restoreState() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (!saved) return;
      if (Number(saved.revision) === revision) {
        pending = Array.isArray(saved.pending) ? saved.pending : [];
        deleted = new Set(Array.isArray(saved.deleted) ? saved.deleted.map(String) : []);
      }
      sectorLayout = String(saved.sectorLayout || '1x1');
      sectorIndex = Number(saved.sectorIndex || 0);
      completedSectors = new Set(Array.isArray(saved.completedSectors) ? saved.completedSectors.map(Number) : []);
      magnifierEnabled = saved.magnifierEnabled !== false;
      sectorSelect.value = sectorLayout;
      if (saved.savedAt) saveLabel.textContent = `복원됨 ${new Date(saved.savedAt).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;
    } catch (_) {}
  }

  function layoutShape() { const [cols,rows] = sectorLayout.split('x').map(Number); return {cols,rows,total:cols*rows}; }
  function sectorBounds() {
    const {cols,rows,total} = layoutShape(); sectorIndex = Math.max(0, Math.min(total-1, sectorIndex));
    const col = sectorIndex % cols, row = Math.floor(sectorIndex / cols);
    const sw = image.naturalWidth/cols, sh = image.naturalHeight/rows;
    return {x:col*sw,y:row*sh,w:sw,h:sh};
  }
  function updateSectorStatus() {
    const {total}=layoutShape();
    sectorProgress.textContent = total===1 ? '전체 보기' : `섹터 ${sectorIndex+1}/${total} · 완료 ${completedSectors.size}`;
    shell.querySelector('[data-action="prev-sector"]').disabled = total===1;
    shell.querySelector('[data-action="next-sector"]').disabled = total===1;
  }

  function resizeCanvas() {
    const rect=stage.getBoundingClientRect(); dpr=Math.max(1,window.devicePixelRatio||1);
    canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr)); draw();
  }
  function fitCurrent() {
    const rect=stage.getBoundingClientRect(); if(!image.naturalWidth||!rect.width) return;
    const b=sectorBounds(); const pad=0.04;
    scale=Math.min(rect.width/(b.w*(1+2*pad)),rect.height/(b.h*(1+2*pad)));
    scale=Math.max(scale,.02);
    offsetX=(rect.width-b.w*scale)/2-b.x*scale;
    offsetY=(rect.height-b.h*scale)/2-b.y*scale;
    draw();
  }
  function imageToScreen(point){return{x:point.x*scale+offsetX,y:point.y*scale+offsetY};}
  function screenToImage(x,y){return{x:Math.max(0,Math.min(image.naturalWidth-1,(x-offsetX)/scale)),y:Math.max(0,Math.min(image.naturalHeight-1,(y-offsetY)/scale))};}
  function linePoints(line){return{p1:line.p1||[line.x1,line.y1],p2:line.p2||[line.x2,line.y2]};}
  function lineDistance(px,py,line){
    const {p1,p2}=linePoints(line); const a=imageToScreen({x:+p1[0],y:+p1[1]}),b=imageToScreen({x:+p2[0],y:+p2[1]});
    const vx=b.x-a.x,vy=b.y-a.y,wx=px-a.x,wy=py-a.y,den=vx*vx+vy*vy;
    const t=den>0?Math.max(0,Math.min(1,(wx*vx+wy*vy)/den)):0;
    return Math.hypot(px-(a.x+t*vx),py-(a.y+t*vy));
  }
  function widthText(line){
    const {p1,p2}=linePoints(line); const aw=Math.hypot(+p2[0]-+p1[0],+p2[1]-+p1[1]);
    const ow=line.width_original_px!=null?+line.width_original_px:aw/analysisScale;
    return nmPerPx!=null&&Number.isFinite(nmPerPx)?`${(ow*nmPerPx).toFixed(2)} nm`:`${ow.toFixed(2)} px`;
  }
  function lineColor(line){return line.source==='manual'?'#16d9e8':(line.source==='orientation'?'#a6ecff':'#ffd54a');}
  function drawPath(line,color){
    const pts=Array.isArray(line.path_points)?line.path_points:[]; if(pts.length<2)return;
    ctx.save();ctx.strokeStyle=color;ctx.lineWidth=4;ctx.shadowColor='rgba(0,0,0,.8)';ctx.shadowBlur=3;ctx.beginPath();
    pts.forEach((p,i)=>{const q=imageToScreen({x:+p[0],y:+p[1]});if(i===0)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y);});ctx.stroke();ctx.restore();
  }
  function drawLine(line,color,lineWidth,dashed=false,label=true,selected=false){
    const {p1,p2}=linePoints(line); const a=imageToScreen({x:+p1[0],y:+p1[1]}),b=imageToScreen({x:+p2[0],y:+p2[1]});
    ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dashed?[7,5]:[]);if(selected){ctx.shadowColor='#fff';ctx.shadowBlur=5;}
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.fillStyle=color;
    for(const p of[a,b]){ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2.4,lineWidth+1),0,Math.PI*2);ctx.fill();}
    if(label){const text=String(line.label??'');const mx=(a.x+b.x)/2,my=(a.y+b.y)/2;ctx.font=selected?'bold 14px sans-serif':'bold 11px sans-serif';const tw=ctx.measureText(text).width;ctx.fillStyle='rgba(0,0,0,.78)';ctx.fillRect(mx-tw/2-3,my-18,tw+6,16);ctx.fillStyle='#fff';ctx.fillText(text,mx-tw/2,my-6);}
    ctx.restore();
  }
  function selectedLine(){return committed.find(line=>String(line.id)===String(selectedId))||null;}
  function updateSelectionPanel(){
    const line=selectedLine(); if(!line){selectionPanel.style.display='none';return;}
    const direction=line.direction_deg==null?'—':`${Number(line.direction_deg).toFixed(1)}°`;
    selectionPanel.innerHTML=`<b>Fiber ${line.label??''}</b><br>두께 ${widthText(line)}<br>해당 위치 방향 ${direction}<br><span style="opacity:.75">클릭한 fiber 중심선을 강조했습니다.</span>`;
    selectionPanel.style.display='block';
  }
  function drawMagnifier(){
    if(!magnifierEnabled||!pointerImage||!image.complete){magnifier.style.display='none';return;}
    magnifier.style.display='block';const size=54;const sx=Math.max(0,Math.min(image.naturalWidth-size,pointerImage.x-size/2));const sy=Math.max(0,Math.min(image.naturalHeight-size,pointerImage.y-size/2));
    mctx.clearRect(0,0,magnifier.width,magnifier.height);mctx.imageSmoothingEnabled=false;mctx.drawImage(image,sx,sy,size,size,0,0,magnifier.width,magnifier.height);
    mctx.strokeStyle='rgba(255,255,255,.95)';mctx.lineWidth=1;mctx.beginPath();mctx.moveTo(magnifier.width/2-12,magnifier.height/2);mctx.lineTo(magnifier.width/2+12,magnifier.height/2);mctx.moveTo(magnifier.width/2,magnifier.height/2-12);mctx.lineTo(magnifier.width/2,magnifier.height/2+12);mctx.stroke();
  }
  function draw(){
    const rect=stage.getBoundingClientRect();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);ctx.fillStyle='#0b0d10';ctx.fillRect(0,0,rect.width,rect.height);
    if(image.complete&&image.naturalWidth){ctx.imageSmoothingEnabled=scale<1;ctx.drawImage(image,offsetX,offsetY,image.naturalWidth*scale,image.naturalHeight*scale);}
    const selected=selectedLine();
    for(const line of committed){
      const ids=Array.isArray(line.erase_ids)?line.erase_ids.map(String):[];const isDeleted=ids.some(id=>deleted.has(id));const isSelected=selected&&String(line.id)===String(selected.id);
      ctx.globalAlpha=selected&&!isSelected?.32:1;
      if(isSelected)drawPath(line,'#00fff0');
      drawLine(line,isDeleted?'rgba(255,72,86,.95)':lineColor(line),isSelected?4.2:(line.source==='manual'?2.8:2.2),isDeleted,true,isSelected);
    }
    ctx.globalAlpha=1;pending.forEach((line,i)=>drawLine({...line,label:`+${i+1}`},'#44a3ff',2.8,true,true,false));
    if(firstPoint){const p=imageToScreen(firstPoint);ctx.fillStyle='#44a3ff';ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke();}
    updateStatus();updateSelectionPanel();drawMagnifier();
  }
  function updateStatus(){
    const labels={pan:'이동·선택 · 드래그로 이동, 선 클릭으로 강조',add:firstPoint?'반대쪽 edge를 클릭하세요':'fiber의 첫 번째 edge를 클릭하세요',erase:'지울 두께선을 클릭하세요'};
    modeLabel.textContent=labels[mode];hint.textContent=labels[mode];countsLabel.textContent=`임시 측정 ${pending.length}개 · 삭제 예정 ${deleted.size}개`;applyButton.disabled=pending.length===0&&deleted.size===0;updateSectorStatus();
  }
  function setMode(next){mode=next;firstPoint=null;shell.querySelectorAll('[data-tool]').forEach(btn=>btn.classList.toggle('active',btn.dataset.tool===mode));stage.style.cursor=mode==='pan'?'grab':(mode==='erase'?'not-allowed':'crosshair');draw();}
  function pointerPosition(event){const rect=canvas.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top};}
  function nearestCommitted(p){let best=null;committed.forEach((line,index)=>{const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={line,index,distance:d};});return best;}

  function onPointerDown(event){stage.focus();const p=pointerPosition(event);moved=false;
    if(mode==='pan'||event.button===1||event.button===2){dragging=true;dragStart=p;dragOrigin={x:offsetX,y:offsetY};stage.setPointerCapture(event.pointerId);stage.style.cursor='grabbing';return;}
    if(event.button!==0)return;
    if(mode==='add'){const point=screenToImage(p.x,p.y);if(!firstPoint)firstPoint=point;else{const d=Math.hypot(point.x-firstPoint.x,point.y-firstPoint.y);if(d>=.75){snapshot();pending.push({p1:[firstPoint.x,firstPoint.y],p2:[point.x,point.y]});queueSave();}firstPoint=null;}draw();return;}
    if(mode==='erase'){let best=null;pending.forEach((line,index)=>{const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={kind:'pending',index,distance:d};});committed.forEach((line,index)=>{const d=lineDistance(p.x,p.y,line);if(!best||d<best.distance)best={kind:'committed',index,distance:d};});if(best&&best.distance<=14){snapshot();if(best.kind==='pending')pending.splice(best.index,1);else{const ids=Array.isArray(committed[best.index].erase_ids)?committed[best.index].erase_ids.map(String):[];const already=ids.length&&ids.every(id=>deleted.has(id));ids.forEach(id=>already?deleted.delete(id):deleted.add(id));}draw();queueSave();}}
  }
  function onPointerMove(event){const p=pointerPosition(event);pointerImage=screenToImage(p.x,p.y);if(dragging){if(Math.hypot(p.x-dragStart.x,p.y-dragStart.y)>3)moved=true;offsetX=dragOrigin.x+(p.x-dragStart.x);offsetY=dragOrigin.y+(p.y-dragStart.y);}draw();}
  function onPointerUp(event){if(!dragging)return;const p=pointerPosition(event);dragging=false;try{stage.releasePointerCapture(event.pointerId);}catch(_){}stage.style.cursor=mode==='pan'?'grab':(mode==='erase'?'not-allowed':'crosshair');if(mode==='pan'&&!moved){const best=nearestCommitted(p);selectedId=best&&best.distance<=15?String(best.line.id):null;}draw();}
  function onPointerLeave(){pointerImage=null;drawMagnifier();}
  function onWheel(event){event.preventDefault();const p=pointerPosition(event),before=screenToImage(p.x,p.y),factor=Math.exp(-event.deltaY*.0015);scale=Math.max(.05,Math.min(28,scale*factor));offsetX=p.x-before.x*scale;offsetY=p.y-before.y*scale;draw();}
  function onKeyDown(event){if(event.key==='Escape'){firstPoint=null;selectedId=null;draw();}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();restoreLast();}if(event.key==='1')setMode('pan');if(event.key==='2')setMode('add');if(event.key==='3')setMode('erase');}

  shell.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>setMode(button.dataset.tool));
  shell.querySelector('[data-action="undo"]').onclick=restoreLast;
  shell.querySelector('[data-action="clear"]').onclick=()=>{if(!pending.length&&!firstPoint)return;snapshot();pending=[];firstPoint=null;draw();queueSave();};
  shell.querySelector('[data-action="fit"]').onclick=fitCurrent;
  magnifierButton.onclick=()=>{magnifierEnabled=!magnifierEnabled;magnifierButton.classList.toggle('active',magnifierEnabled);magnifierButton.textContent=magnifierEnabled?'⌕ 돋보기 ON':'⌕ 돋보기 OFF';draw();queueSave();};
  sectorSelect.onchange=()=>{sectorLayout=sectorSelect.value;sectorIndex=0;completedSectors=new Set();fitCurrent();queueSave();};
  shell.querySelector('[data-action="prev-sector"]').onclick=()=>{const {total}=layoutShape();sectorIndex=(sectorIndex-1+total)%total;fitCurrent();queueSave();};
  shell.querySelector('[data-action="next-sector"]').onclick=()=>{const {total}=layoutShape();completedSectors.add(sectorIndex);sectorIndex=(sectorIndex+1)%total;fitCurrent();queueSave();};
  applyButton.onclick=()=>{if(!pending.length&&!deleted.size)return;saveState();applyButton.disabled=true;setTriggerValue('apply',{new_measurements:pending.map(line=>({p1:line.p1,p2:line.p2})),delete_ids:Array.from(deleted)});};

  stage.addEventListener('pointerdown',onPointerDown);stage.addEventListener('pointermove',onPointerMove);stage.addEventListener('pointerup',onPointerUp);stage.addEventListener('pointercancel',onPointerUp);stage.addEventListener('pointerleave',onPointerLeave);stage.addEventListener('wheel',onWheel,{passive:false});stage.addEventListener('keydown',onKeyDown);stage.addEventListener('contextmenu',e=>e.preventDefault());
  const observer=new ResizeObserver(resizeCanvas);observer.observe(stage);restoreState();
  const autosaveInterval=setInterval(saveState,300000);
  image.onload=()=>{resizeCanvas();fitCurrent();};image.src=String(data.image_url||'');setMode('add');magnifierButton.classList.toggle('active',magnifierEnabled);magnifierButton.textContent=magnifierEnabled?'⌕ 돋보기 ON':'⌕ 돋보기 OFF';

  const cleanup=()=>{observer.disconnect();clearInterval(autosaveInterval);clearTimeout(saveTimer);stage.removeEventListener('pointerdown',onPointerDown);stage.removeEventListener('pointermove',onPointerMove);stage.removeEventListener('pointerup',onPointerUp);stage.removeEventListener('pointercancel',onPointerUp);stage.removeEventListener('pointerleave',onPointerLeave);stage.removeEventListener('wheel',onWheel);stage.removeEventListener('keydown',onKeyDown);};
  shell.__vfCleanup=cleanup;return cleanup;
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
    autosave_key: str | None = None,
):
    """Render the browser-local review canvas.

    Zoom, sectors, magnifier, highlighting and five-minute autosave stay in the
    browser. Python reruns only after ``전체 반영``.
    """
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
            "autosave_key": str(autosave_key or key),
        },
        key=key,
        on_apply_change=lambda: None,
    )
