# Interactive Review and Collaboration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend VisionFlux with a cursor-following magnifier, detected-path hover guidance, manual thickness editing with normal guides, label controls, 3D thickness-direction counts, dual annotated exports, and optional Supabase team sharing while preserving the current detector and Streamlit workflow.

**Architecture:** Keep the expensive image analysis unchanged. Add all pointer interactions to the browser-side custom canvas so mouse movement never reruns Python. Export and collaboration stay in focused Python modules; Supabase remains optional and is activated only when secrets are configured.

**Tech Stack:** Python 3.11, Streamlit components v2, JavaScript Canvas 2D, NumPy/Pandas/Pillow/Plotly, optional supabase-py.

## Global Constraints

- Hover highlighting must only select fibers with a detected `path_points` centerline.
- Missing/unrecognized fibers may still be manually measured, but receive no automatic path highlight or normal guide.
- Normal guidance uses detected path tangent first, then stored local direction, and disappears when neither is reliable.
- Existing zoom, sector review, eraser, undo, five-minute browser autosave, email, batch upload, and detector behavior remain available.
- Supabase must be optional; the app must run without collaboration secrets.
- Secrets must never be committed.

---

### Task 1: Canvas interaction model

**Files:**
- Modify: `ui/measurement_canvas.py`
- Test: `tests/test_measurement_canvas.py`

**Interfaces:**
- Consumes: representative line objects with `id`, `path_points`, `direction_deg`, `erase_ids`.
- Produces: the existing `apply` payload containing `new_measurements` and `delete_ids`.

- [ ] Add failing source-level tests for edit mode, moving magnifier, label toggle, 1.5-second detected-path hover, magnifier overlays, and normal-guide preview.
- [ ] Implement cursor-following magnifier with edge-aware placement.
- [ ] Render committed, pending, selected paths, selected lines, edit guides, and the first click inside the magnifier.
- [ ] Add delayed hover selection restricted to lines with at least two `path_points`.
- [ ] Add `두께 수정` mode that replaces a representative line by deleting its source IDs and adding a corrected manual chord.
- [ ] Add detected-path normal guidance after the first edge click; keep manual second-point selection when no detected path is nearby.
- [ ] Add label display ON/OFF without modifying stored labels.
- [ ] Run Python tests and Node syntax validation.

### Task 2: Export variants

**Files:**
- Modify: `pipeline/exports.py`
- Modify: `pipeline/review.py`
- Modify: `app.py`
- Test: `tests/test_exports.py`

**Interfaces:**
- Produces: `ExportBundle.annotated_labeled_png` and `ExportBundle.annotated_unlabeled_png`.

- [ ] Add failing tests proving labeled and unlabeled PNGs are both produced and differ.
- [ ] Parameterize image rendering with `show_labels`.
- [ ] Include both images in the result ZIP and expose separate download buttons.
- [ ] Preserve the ImageJ-like CSV schema and intensity semantics.

### Task 3: Thickness-direction-count visualization

**Files:**
- Modify: `ui/figures.py`
- Modify: `app.py`
- Test: `tests/test_ui_helpers.py`

**Interfaces:**
- Consumes: final representative canvas lines and calibrated/uncalibrated thickness.
- Produces: Plotly 3D surface with direction bins, thickness bins, and counts.

- [ ] Add a failing test for bin counts and axis titles.
- [ ] Build a robust 2D histogram and 3D surface; return an empty-state figure when data are insufficient.
- [ ] Add the figure below the regular thickness histogram.

### Task 4: Optional Supabase collaboration

**Files:**
- Create: `services/collaboration.py`
- Create: `supabase_schema.sql`
- Modify: `app.py`
- Modify: `requirements.txt`
- Modify: `.streamlit/secrets.toml.example`
- Modify: `README.md`
- Test: `tests/test_collaboration.py`

**Interfaces:**
- Produces: `CollaborationConfig`, snapshot serialization/deserialization, optional save/list/load operations.
- Uses: `visionflux_reviews` table keyed by `(project_id, image_hash)`.

- [ ] Add failing tests for config parsing, deterministic snapshot JSON, and applying a loaded snapshot to a current `ReviewItem`.
- [ ] Implement lazy Supabase import and optional client creation.
- [ ] Implement project dashboard records, worker claim/status, upsert snapshot, list snapshots, and load snapshot.
- [ ] Add sidebar collaboration UI that is hidden when secrets are absent.
- [ ] Save current committed review data; allow another worker who analyzes the same image to load the shared snapshot.
- [ ] Document Supabase table setup and Streamlit secrets.

### Task 5: Full verification and artifact

**Files:**
- Modify: `README.md`
- Create: final ZIP artifact.

- [ ] Run the entire pytest suite.
- [ ] Run `python -m compileall` on the project.
- [ ] Validate embedded JavaScript with Node.
- [ ] Run a synthetic detector/analyzer smoke test.
- [ ] Build and verify the ZIP contents.
