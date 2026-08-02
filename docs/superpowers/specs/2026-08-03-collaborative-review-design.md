# VisionFlux Collaborative Review Design

## Goal
Keep the fast detector and existing review workflow while adding a mouse-following magnifier, recognized-path-only hover highlighting, guided add/modify tools, label toggles, 3D thickness-direction counts, paired image exports, and optional Supabase collaboration for five reviewers.

## Interaction rules
- Hover highlighting runs only for model-recognized lines with at least two `path_points`; manual or missing paths are never inferred or highlighted.
- In add mode, a 1.5 second stable hover highlights the nearest recognized path. The first edge click uses that path tangent to show a normal guide; when no recognized path is close, no guide is shown and the second edge is freehand.
- In modify mode, selecting an existing endpoint shows the existing chord direction as an infinite guide. The replacement endpoint is projected onto that guide and applied as a manual replacement plus deletion of the old representative group.
- The magnifier follows the pointer, flips away from viewport edges, and redraws all visible overlays.
- Labels can be hidden without changing the stable internal IDs. Visible labels are regenerated without gaps after apply.

## Collaboration
- Supabase is optional. Without secrets, local upload/review works exactly as before.
- A server-side service-role key is kept only in Streamlit Secrets. Tables use RLS with no public policies; Storage is private.
- Original images live in Storage. Shared review snapshots contain measurements, feedback, calibration, revision, canvas pending state, and sector progress.
- An atomic SQL RPC acquires a time-limited image lock. Other users can load a locked image read-only.
- The browser keeps local autosave and emits a server autosave event every five minutes. The app writes that snapshot to Supabase.

## Exports
- ImageJ-compatible CSV columns remain `label, Area, Mean, Min, Max, Angle, Length`; intensity statistics are sampled along the line ROI.
- Export both labeled and unlabeled PNGs.
- Add a direction matching CSV and a 3D binned count plot using one representative line per fiber region.
