from __future__ import annotations

import time

import streamlit.components.v1 as components


def live_elapsed_timer(started_at_unix: float | None = None) -> None:
    """Render a browser-side elapsed timer that keeps ticking while Python is busy."""
    started_ms = int(1000 * (time.time() if started_at_unix is None else float(started_at_unix)))
    components.html(
        f"""
<div id="vf-elapsed" style="font: 13px system-ui; color: #aeb8c5; padding: 2px 0;">
  경과 시간 0분 00초
</div>
<script>
(() => {{
  const start = {started_ms};
  const target = document.getElementById('vf-elapsed');
  const format = (seconds) => {{
    const total = Math.max(0, Math.floor(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0
      ? `${{h}}시간 ${{m}}분 ${{String(s).padStart(2, '0')}}초`
      : `${{m}}분 ${{String(s).padStart(2, '0')}}초`;
  }};
  const tick = () => {{
    target.textContent = `경과 시간 ${{format(Date.now() / 1000 - start / 1000)}}`;
  }};
  tick();
  const timer = setInterval(tick, 500);
  window.addEventListener('beforeunload', () => clearInterval(timer), {{once: true}});
}})();
</script>
""",
        height=34,
        scrolling=False,
    )
