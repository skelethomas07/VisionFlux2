# VisionFlux Collaborative Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add guided review interactions, dual exports, 3D distributions, and optional Supabase sharing while retaining the existing fast analysis pipeline.

**Architecture:** Extend the browser-side Streamlit component for latency-sensitive interactions. Keep Python as the source of truth after Apply and for exports. Add a focused Supabase repository and snapshot serializer, with local mode remaining the default fallback.

**Tech Stack:** Python 3.11+, Streamlit 1.60 components v2, NumPy/Pandas/Plotly, supabase-py, PostgreSQL SQL migration.

## Global Constraints
- Never highlight a fiber without model-provided `path_points`.
- Do not expose a Supabase service-role key to browser JavaScript or GitHub.
- Preserve current local upload, email, batch progress, detector, and review behavior.
- Server autosave interval is 300 seconds.

### Task 1: Canvas payload and guided interactions
- [ ] Add replacement-compatible payload normalization and canvas state.
- [ ] Add moving magnifier with overlay rendering.
- [ ] Add stable-hover recognized-path selection and normal guide.
- [ ] Add modify tool, label toggle, and autosave trigger.
- [ ] Test payload normalization and static JS syntax.

### Task 2: Export and visual analysis
- [ ] Emit labeled and unlabeled PNGs.
- [ ] Add 3D thickness-direction count and heatmap figures.
- [ ] Update session ZIP and download buttons.
- [ ] Test output bytes and histogram counts.

### Task 3: Supabase collaboration
- [ ] Add migration, private buckets, and atomic lock RPCs.
- [ ] Add configuration, repository, snapshot serialization, and lock helpers.
- [ ] Add optional shared-project UI and five-minute server autosave.
- [ ] Test serialization and repository behavior with fakes.

### Task 4: Packaging and verification
- [ ] Update requirements, secrets example, README, and GitHub workflow.
- [ ] Run compile, tests, JS syntax validation, synthetic analysis, and ZIP integrity checks.
