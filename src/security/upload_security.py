"""Upload validation, malware hooks, and audit logging."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import tempfile
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import HTTPException

from src.tracing import REQUEST_ID_HEADER

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_AUDIT_FILE = PROJECT_ROOT / "data" / "audit" / "uploads.jsonl"
_audit_lock = Lock()

ALLOWED_UPLOADS: dict[str, dict[str, Any]] = {
    "pdf": {
        "content_types": {"application/pdf"},
        "extensions": {".pdf"},
    },
    "docx": {
        "content_types": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
        "extensions": {".docx"},
    },
    "image": {
        "content_types": {"image/jpeg", "image/png", "image/webp"},
        "extensions": {".jpg", ".jpeg", ".png", ".webp"},
    },
}


@dataclass(frozen=True)
class UploadSecurityMetadata:
    filename: str
    content_type: str
    kind: str
    size_bytes: int
    sha256: str

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def sanitize_upload_filename(filename: str | None) -> str:
    cleaned = Path((filename or "upload").replace("\x00", "")).name.strip()
    return cleaned or "upload"


def classify_upload(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    matches = []
    for kind, spec in ALLOWED_UPLOADS.items():
        if content_type in spec["content_types"] and suffix in spec["extensions"]:
            matches.append(kind)
    if len(matches) == 1:
        return matches[0]
    raise HTTPException(
        status_code=415,
        detail="Unsupported upload type. Only PDF, DOCX, JPEG, PNG, and WebP files are allowed.",
    )


async def validate_upload_security(
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    expected_kind: str | None = None,
) -> UploadSecurityMetadata:
    safe_filename = sanitize_upload_filename(filename)
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    kind = classify_upload(safe_filename, normalized_content_type)
    if expected_kind and kind != expected_kind:
        raise HTTPException(status_code=415, detail=f"Expected {expected_kind} upload.")
    if not data:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {safe_filename}")

    _validate_magic_bytes(kind, data, safe_filename)
    _reject_malicious_document(kind, data, safe_filename)
    await _scan_with_configured_av(data, safe_filename)

    return UploadSecurityMetadata(
        filename=safe_filename,
        content_type=normalized_content_type,
        kind=kind,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _validate_magic_bytes(kind: str, data: bytes, filename: str) -> None:
    if kind == "pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail=f"File content is not a valid PDF: {filename}")
    if kind == "docx" and not data.startswith(b"PK\x03\x04"):
        raise HTTPException(status_code=415, detail=f"File content is not a valid DOCX: {filename}")
    if kind == "image":
        suffix = Path(filename).suffix.lower()
        valid = (
            suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8\xff")
        ) or (
            suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n")
        ) or (
            suffix == ".webp" and len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        )
        if not valid:
            raise HTTPException(
                status_code=415,
                detail=f"Image extension, content type, and file signature do not match: {filename}",
            )


def _reject_malicious_document(kind: str, data: bytes, filename: str) -> None:
    if kind == "pdf":
        lowered = data[:2_000_000].lower()
        if b"/javascript" in lowered or b"/js" in lowered or b"/openaction" in lowered:
            raise HTTPException(
                status_code=422,
                detail=f"PDF contains active content and was rejected: {filename}",
            )
    elif kind == "docx":
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                _validate_zip_archive(archive, filename)
                names = {info.filename for info in archive.infolist()}
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=415, detail=f"Invalid DOCX archive: {filename}") from exc
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise HTTPException(status_code=415, detail=f"Invalid DOCX structure: {filename}")
        if any(name.lower().endswith("vbaproject.bin") for name in names):
            raise HTTPException(
                status_code=422,
                detail=f"Macro-enabled Office documents are not allowed: {filename}",
            )


def _validate_zip_archive(archive: zipfile.ZipFile, filename: str) -> None:
    total_uncompressed = 0
    max_uncompressed = _env_int("UPLOAD_MAX_UNCOMPRESSED_MB", 100, 1, 500) * 1024 * 1024
    max_ratio = _env_int("UPLOAD_MAX_COMPRESSION_RATIO", 100, 1, 10_000)
    for info in archive.infolist():
        name = info.filename
        parts = Path(name).parts
        if name.startswith("/") or ".." in parts:
            raise HTTPException(status_code=422, detail=f"Archive path traversal rejected: {filename}")
        total_uncompressed += info.file_size
        if total_uncompressed > max_uncompressed:
            raise HTTPException(status_code=413, detail=f"Archive expands too large: {filename}")
        if info.compress_size and info.file_size / info.compress_size > max_ratio:
            raise HTTPException(status_code=422, detail=f"Suspicious archive compression: {filename}")


async def _scan_with_configured_av(data: bytes, filename: str) -> None:
    command = os.getenv("UPLOAD_AV_COMMAND", "").strip()
    if not command:
        logger.debug("UPLOAD_AV_COMMAND not configured; skipping external antivirus scan")
        return

    timeout_s = float(_env_int("UPLOAD_AV_TIMEOUT_SECONDS", 30, 1, 300))
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="gaokao-upload-", suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name
        args = [*shlex.split(command), temp_path]
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise HTTPException(status_code=503, detail="Antivirus scan timed out.") from exc
        if process.returncode != 0:
            logger.warning(
                "Antivirus rejected upload",
                extra={
                    "filename": filename,
                    "returncode": process.returncode,
                    "stdout": stdout.decode(errors="replace")[:500],
                    "stderr": stderr.decode(errors="replace")[:500],
                },
            )
            raise HTTPException(status_code=422, detail=f"Upload failed antivirus scan: {filename}")
    finally:
        if temp_path:
            try:
                await asyncio.to_thread(Path(temp_path).unlink, missing_ok=True)
            except OSError:
                logger.warning("Failed to remove upload scan temp file", extra={"path": temp_path})


def record_upload_audit(
    *,
    action: str,
    status: str,
    files: list[UploadSecurityMetadata] | list[dict[str, Any]],
    request: Any | None = None,
    task_id: str | None = None,
    error: str | None = None,
) -> None:
    entry = {
        "ts": time.time(),
        "action": action,
        "status": status,
        "task_id": task_id,
        "request_id": request.headers.get(REQUEST_ID_HEADER) if request is not None else None,
        "username": getattr(getattr(request, "state", None), "username", None) if request is not None else None,
        "client": request.client.host if request is not None and request.client else None,
        "error": error,
        "files": [
            item.to_audit_dict() if isinstance(item, UploadSecurityMetadata) else item
            for item in files
        ],
    }
    try:
        UPLOAD_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _audit_lock, UPLOAD_AUDIT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        logger.warning("Failed to write upload audit log", exc_info=True)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
