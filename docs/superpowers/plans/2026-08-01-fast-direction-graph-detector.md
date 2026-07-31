# Fast Direction Graph Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slow legacy beam-search detector with a coarse-to-fine direction graph detector that finds more fibers, handles curved fibers with local direction segments, and uses a conservative pore mask to avoid merging adjacent fibers.

**Architecture:** Compute normalized image, gradient, structure tensor, multiscale Hessian ridge response, and pore mask once. Skeletonize a high-recall ridge mask, split it into graph paths, measure thickness in vectorized normal profiles, and group local directions into contiguous segments. Keep the existing Streamlit review, batch, email, export, and manual correction interfaces unchanged.

**Tech Stack:** Python, NumPy, SciPy, scikit-image, pandas, optional CuPy, Streamlit.

## Global Constraints

- Keep the current `AnalysisResult` schema and review UI compatible.
- Keep `pipeline/legacy_pipeline.py` untouched as an emergency fallback.
- Default to the new fast detector on Streamlit Community Cloud CPU.
- Use GPU only for array-heavy tensor/ridge operations when CuPy is actually available.
- Preserve manual measurement, batch upload, email, progress, and export features.

---

### Task 1: Fast detector core
- [ ] Add tests for curved fibers, pore-separated fibers, schema, and speed.
- [ ] Implement one-pass feature maps, pore core, skeleton paths, direction segments, and vectorized width measurement.
- [ ] Verify synthetic detection quality and runtime.

### Task 2: Analyzer integration
- [ ] Add tests proving the fast detector is the default and progress stays monotonic.
- [ ] Replace the legacy primary call with fast detector output, retaining legacy fallback.
- [ ] Reuse the same orientation result instead of calculating it twice.

### Task 3: Direction-segment UI
- [ ] Add a length-weighted direction-segment plot test.
- [ ] Display curved-fiber local direction segments rather than one direction per whole fiber.
- [ ] Add summary fields for path and segment counts.

### Task 4: Verification and packaging
- [ ] Run the full test suite.
- [ ] Run a synthetic end-to-end benchmark.
- [ ] Compile all Python modules.
- [ ] Build a GitHub upload ZIP.
