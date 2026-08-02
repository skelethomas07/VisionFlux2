# VisionFlux Batch Progress, Email, and GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing VisionFlux Streamlit app with multi-image analysis, visible progress and elapsed time, optional completion email, and automatic GPU acceleration where supported, while preserving the current analysis and review workflow.

**Architecture:** Keep `run_uploaded_analysis` as the single-image entry point and add a lightweight batch coordinator around it. Store review state per image in Streamlit session state, expose one selected image at a time to the existing thickness/orientation UI, and use optional CuPy only for OrientationJ structure-tensor filtering while retaining CPU fallback for every environment. Email uses Gmail SMTP credentials from Streamlit Secrets; the user enters only the recipient address.

**Tech Stack:** Python 3.12, Streamlit 1.60, NumPy/SciPy/scikit-image, optional CuPy, smtplib, pytest.

## Global Constraints

- Existing automatic fiber detection and interactive review behavior must remain available.
- Multiple uploaded images are analyzed sequentially to avoid memory spikes.
- Email failures must never discard successful analysis results.
- Secrets must not be committed to GitHub.
- GPU support must be optional and fall back to CPU without raising an application error.
- Streamlit Community Cloud must remain installable from `requirements.txt` without CUDA packages.

---

### Task 1: Batch and progress primitives

**Files:**
- Create: `pipeline/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Produces: `format_elapsed(seconds) -> str`, `BatchInput`, `BatchOutcome`, and `run_batch(inputs, analyze, on_progress) -> list[BatchOutcome]`.

- [ ] Write tests for elapsed-time formatting, ordered multi-file execution, progress mapping, and per-file failure isolation.
- [ ] Run `pytest tests/test_batch.py -v` and verify failure because the module is missing.
- [ ] Implement the minimal batch coordinator and progress event dataclass.
- [ ] Run `pytest tests/test_batch.py -v` and verify all tests pass.

### Task 2: Completion email service

**Files:**
- Create: `services/__init__.py`
- Create: `services/notifications.py`
- Create: `.streamlit/secrets.toml.example`
- Test: `tests/test_notifications.py`

**Interfaces:**
- Produces: `EmailConfig`, `CompletionReport`, `validate_email_address`, `send_completion_email`.

- [ ] Write tests for address validation, message contents, SMTP SSL login/send, and missing configuration.
- [ ] Run `pytest tests/test_notifications.py -v` and verify failure because the service is missing.
- [ ] Implement Gmail SMTP sending with injectable SMTP factory for tests.
- [ ] Run `pytest tests/test_notifications.py -v` and verify all tests pass.

### Task 3: Optional GPU backend and analyzer progress

**Files:**
- Create: `pipeline/compute.py`
- Modify: `pipeline/orientation.py`
- Modify: `pipeline/analyzer.py`
- Create: `requirements-gpu.txt`
- Test: `tests/test_compute.py`
- Modify: `tests/test_orientation.py`
- Modify: `tests/test_analyzer.py`

**Interfaces:**
- Produces: `detect_compute_backend(prefer_gpu=True)`, `gaussian_filter_numpy(...)`.
- Extends: `analyze_orientation(..., prefer_gpu=True)` and `run_uploaded_analysis(..., prefer_gpu=True, progress_callback=None)`.

- [ ] Write tests proving CPU fallback, optional GPU selection through a fake backend, NumPy outputs, and monotonic analyzer progress callbacks.
- [ ] Run the focused tests and verify they fail for missing interfaces.
- [ ] Implement backend detection and CuPy Gaussian filtering with conversion back to NumPy.
- [ ] Add stage progress callbacks to the analyzer without changing its result schema.
- [ ] Run focused tests and verify they pass.

### Task 4: Streamlit multi-image workflow

**Files:**
- Modify: `app.py`
- Create: `ui/live_timer.py`
- Test: `tests/test_app_state.py`
- Modify: `tests/test_imports.py`

**Interfaces:**
- Session state stores `batch_items`, `selected_item_id`, and `last_batch_report`.
- The existing thickness and orientation tabs consume the selected item state.

- [ ] Write tests for item-state creation, image selection, and recomputation isolation.
- [ ] Run focused tests and verify failure because batch state helpers are missing.
- [ ] Replace the single-file uploader with `accept_multiple_files=True`.
- [ ] Map per-image stage progress to overall percentage and show filename, count, elapsed time, and backend.
- [ ] Send one completion email after all images finish when a valid recipient is entered.
- [ ] Preserve successful results when one image or the email send fails.
- [ ] Run focused tests and verify they pass.

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Documents Cloud secrets, local secrets, optional GPU installation, batch operation, and limitations.

- [ ] Update README with exact Gmail app-password and Streamlit Secrets setup.
- [ ] Add optional GPU installation instructions using `requirements-gpu.txt`.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall -q app.py pipeline services ui`.
- [ ] Build a clean GitHub upload ZIP.
