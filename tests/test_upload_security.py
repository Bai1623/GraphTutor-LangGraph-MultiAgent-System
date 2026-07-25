from __future__ import annotations

import json
import os
import zipfile
import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException

from src.security import upload_security


@pytest.mark.asyncio
async def test_rejects_extension_and_magic_mismatch():
    with pytest.raises(HTTPException) as exc:
        await upload_security.validate_upload_security(
            data=b"not a pdf",
            filename="exam.pdf",
            content_type="application/pdf",
        )

    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_pdf_active_content():
    with pytest.raises(HTTPException) as exc:
        await upload_security.validate_upload_security(
            data=b"%PDF-1.4\n/OpenAction << /S /JavaScript /JS (app.alert('x')) >>",
            filename="exam.pdf",
            content_type="application/pdf",
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_rejects_docx_macros():
    docx = _docx_bytes({"word/vbaProject.bin": b"macro"})

    with pytest.raises(HTTPException) as exc:
        await upload_security.validate_upload_security(
            data=docx,
            filename="exam.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_accepts_clean_docx():
    docx = _docx_bytes()

    metadata = await upload_security.validate_upload_security(
        data=docx,
        filename="../exam.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert metadata.filename == "exam.docx"
    assert metadata.kind == "docx"
    assert metadata.sha256


def test_upload_audit_writes_jsonl(monkeypatch, tmp_path):
    audit_file = tmp_path / "uploads.jsonl"
    monkeypatch.setattr(upload_security, "UPLOAD_AUDIT_FILE", audit_file)

    upload_security.record_upload_audit(
        action="document_parse",
        status="accepted",
        task_id="task-1",
        files=[{
            "filename": "exam.pdf",
            "content_type": "application/pdf",
            "kind": "pdf",
            "size_bytes": 12,
        }],
    )

    payload = json.loads(audit_file.read_text(encoding="utf-8"))
    assert payload["action"] == "document_parse"
    assert payload["status"] == "accepted"
    assert payload["task_id"] == "task-1"
    assert payload["files"][0]["filename"] == "exam.pdf"


@pytest.mark.asyncio
async def test_antivirus_scan_removes_temp_file(monkeypatch, tmp_path):
    seen_paths: list[str] = []

    async def fake_exec(*args, **kwargs):
        seen_paths.append(args[-1])
        return _FakeProcess(returncode=0)

    monkeypatch.setenv("UPLOAD_AV_COMMAND", "fake-av")
    monkeypatch.setattr(upload_security.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(upload_security.asyncio, "create_subprocess_exec", fake_exec)

    await upload_security.validate_upload_security(
        data=b"%PDF-1.4\nbody",
        filename="exam.pdf",
        content_type="application/pdf",
    )

    assert seen_paths
    assert not await asyncio.to_thread(os.path.exists, seen_paths[0])


def _docx_bytes(extra_files: dict[str, bytes] | None = None) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<w:document />")
        for name, data in (extra_files or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


class _FakeProcess:
    def __init__(self, *, returncode: int) -> None:
        self.returncode = returncode

    async def communicate(self):
        return b"ok", b""

    def kill(self) -> None:
        return None
