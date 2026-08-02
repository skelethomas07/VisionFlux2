from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.review_state import recompute_review_item


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    """Convert pandas/NumPy values into deterministic JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_json_value(v) for v in list(value)]
    return str(value) if not isinstance(value, (str, bytes)) else value


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


@dataclass(frozen=True)
class CollaborationConfig:
    url: str
    key: str
    project_id: str = "visionflux-shared"
    table: str = "visionflux_reviews"
    images_table: str = "visionflux_images"
    projects_table: str = "visionflux_projects"
    artifacts_table: str = "visionflux_artifacts"
    images_bucket: str = "visionflux-images"
    results_bucket: str = "visionflux-results"
    lock_timeout_minutes: int = 30

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CollaborationConfig":
        url = str(values.get("url", "")).strip()
        key = str(values.get("service_role_key", values.get("key", ""))).strip()
        if not url or not key:
            raise ValueError("Supabase url과 service_role_key가 필요합니다.")
        return cls(
            url=url,
            key=key,
            project_id=str(values.get("project_id", "visionflux-shared")).strip() or "visionflux-shared",
            table=str(values.get("table", "visionflux_reviews")).strip() or "visionflux_reviews",
            images_table=str(values.get("images_table", "visionflux_images")).strip() or "visionflux_images",
            projects_table=str(values.get("projects_table", "visionflux_projects")).strip() or "visionflux_projects",
            artifacts_table=str(values.get("artifacts_table", "visionflux_artifacts")).strip() or "visionflux_artifacts",
            images_bucket=str(values.get("images_bucket", "visionflux-images")).strip() or "visionflux-images",
            results_bucket=str(values.get("results_bucket", "visionflux-results")).strip() or "visionflux-results",
            lock_timeout_minutes=max(5, int(values.get("lock_timeout_minutes", 30))),
        )


def serialize_snapshot(
    item: Any,
    *,
    worker_name: str,
    status: str = "in_progress",
    canvas_state: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": 2,
        "item_id": str(getattr(item, "item_id", "")),
        "image_name": str(getattr(getattr(item, "analysis", None), "image_name", "")),
        "worker_name": str(worker_name),
        "status": str(status),
        "revision": int(getattr(item, "revision", 0)),
        "nm_per_px": _json_value(getattr(item, "nm_per_px", None)),
        "measurements": _records(getattr(item, "measurements", None)),
        "feedback": _json_value(getattr(item, "feedback", [])),
        "canvas_state": _json_value(dict(canvas_state or {})),
    }
    # Stable output supports deduplication and deterministic tests. Timestamps live
    # in the database row rather than inside the snapshot body.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deserialize_snapshot(snapshot: str | bytes | Mapping[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    if isinstance(snapshot, Mapping):
        data = dict(snapshot)
    else:
        if isinstance(snapshot, bytes):
            snapshot = snapshot.decode("utf-8")
        data = json.loads(snapshot)
    if not isinstance(data, dict):
        raise ValueError("공유 스냅샷은 JSON 객체여야 합니다.")
    return data


def apply_snapshot_to_item(item: Any, snapshot: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    data = deserialize_snapshot(snapshot)
    measurements = data.get("measurements", [])
    item.measurements = pd.DataFrame(measurements)
    item.feedback = list(data.get("feedback", []) or [])
    item.revision = int(data.get("revision", 0) or 0)
    nm_per_px = data.get("nm_per_px")
    item.nm_per_px = None if nm_per_px is None else float(nm_per_px)
    item.last_apply_token = None
    recompute_review_item(item)
    return dict(data.get("canvas_state", {}) or {})


def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collaboration_enabled(values: Mapping[str, Any] | None) -> bool:
    if not values:
        return False
    return bool(str(values.get("url", "")).strip() and str(values.get("service_role_key", values.get("key", ""))).strip())


class SupabaseRepository:
    """Small server-side Supabase repository.

    The service-role key is read from Streamlit Secrets and never sent to the
    custom component. Tables and Storage can therefore remain private with RLS
    enabled and no public policies.
    """

    def __init__(self, config: CollaborationConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            from supabase import create_client

            client = create_client(config.url, config.key)
        self.client = client

    def ensure_project(self, *, name: str | None = None) -> dict[str, Any]:
        slug = self.config.project_id
        query = (
            self.client.table(self.config.projects_table)
            .select("*")
            .eq("slug", slug)
            .limit(1)
            .execute()
        )
        if query.data:
            return dict(query.data[0])
        created = (
            self.client.table(self.config.projects_table)
            .insert({"slug": slug, "name": name or slug})
            .execute()
        )
        if not created.data:
            raise RuntimeError("Supabase 프로젝트 행을 만들지 못했습니다.")
        return dict(created.data[0])

    def list_images(self, project_uuid: str) -> list[dict[str, Any]]:
        response = (
            self.client.table(self.config.images_table)
            .select("*")
            .eq("project_id", project_uuid)
            .order("image_name")
            .execute()
        )
        return [dict(row) for row in (response.data or [])]

    def upload_image(
        self,
        *,
        project_uuid: str,
        filename: str,
        data: bytes,
        uploaded_by: str,
    ) -> dict[str, Any]:
        digest = image_sha256(data)
        safe_name = filename.replace("/", "_").replace("\\", "_")
        path = f"{project_uuid}/originals/{digest[:16]}-{safe_name}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.client.storage.from_(self.config.images_bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        payload = {
            "project_id": project_uuid,
            "image_name": filename,
            "image_hash": digest,
            "storage_path": path,
            "uploaded_by": uploaded_by,
            "updated_at": _utc_now_iso(),
        }
        response = (
            self.client.table(self.config.images_table)
            .upsert(payload, on_conflict="project_id,image_hash")
            .execute()
        )
        if not response.data:
            raise RuntimeError("이미지 메타데이터를 저장하지 못했습니다.")
        return dict(response.data[0])

    def download_image(self, storage_path: str) -> bytes:
        data = self.client.storage.from_(self.config.images_bucket).download(storage_path)
        return bytes(data)

    def acquire_lock(self, image_id: str, worker_name: str) -> dict[str, Any]:
        response = self.client.rpc(
            "visionflux_acquire_lock",
            {
                "p_image_id": image_id,
                "p_worker": worker_name,
                "p_timeout_minutes": self.config.lock_timeout_minutes,
            },
        ).execute()
        rows = response.data or []
        if isinstance(rows, dict):
            return dict(rows)
        return dict(rows[0]) if rows else {"acquired": False, "locked_by": None}

    def release_lock(self, image_id: str, worker_name: str, *, completed: bool = False) -> None:
        self.client.rpc(
            "visionflux_release_lock",
            {"p_image_id": image_id, "p_worker": worker_name, "p_completed": bool(completed)},
        ).execute()

    def save_snapshot(
        self,
        *,
        image_id: str,
        worker_name: str,
        snapshot: str,
        status: str = "in_progress",
    ) -> dict[str, Any]:
        parsed = deserialize_snapshot(snapshot)
        payload = {
            "image_id": image_id,
            "snapshot": parsed,
            "revision": int(parsed.get("revision", 0) or 0),
            "updated_by": worker_name,
            "status": status,
            "updated_at": _utc_now_iso(),
        }
        response = (
            self.client.table(self.config.table)
            .upsert(payload, on_conflict="image_id")
            .execute()
        )
        self.client.table(self.config.images_table).update(
            {"status": status, "updated_at": _utc_now_iso()}
        ).eq("id", image_id).execute()
        return dict(response.data[0]) if response.data else payload

    def load_snapshot(self, image_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table(self.config.table)
            .select("*")
            .eq("image_id", image_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        row = dict(response.data[0])
        row["snapshot"] = deserialize_snapshot(row.get("snapshot"))
        return row

    def upload_artifact(
        self,
        *,
        image_id: str,
        kind: str,
        filename: str,
        data: bytes,
        content_type: str,
        worker_name: str,
    ) -> str:
        safe_name = filename.replace("/", "_").replace("\\", "_")
        path = f"{image_id}/{kind}/{safe_name}"
        self.client.storage.from_(self.config.results_bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        self.client.table(self.config.artifacts_table).insert({
            "image_id": image_id,
            "kind": kind,
            "storage_path": path,
            "created_by": worker_name,
        }).execute()
        return path
