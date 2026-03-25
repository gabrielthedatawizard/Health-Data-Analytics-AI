import io
from uuid import uuid4

from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def _upload_document(actor: str, filename: str, content: str | bytes, content_type: str = "text/plain") -> dict:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    files = {"file": (filename, io.BytesIO(payload), content_type)}
    response = client.post("/documents", files=files, headers={"X-API-Key": actor})
    assert response.status_code == 200
    return response.json()


def test_document_library_is_user_scoped() -> None:
    actor_a = f"doc_scope_a_{uuid4().hex}"
    actor_b = f"doc_scope_b_{uuid4().hex}"

    uploaded_a = _upload_document(actor_a, "policy_a.md", "# Policy A\nDenominator exclusions require supervisor review.")
    uploaded_b = _upload_document(actor_b, "policy_b.md", "# Policy B\nAdmissions should be monitored weekly.")

    list_a = client.get("/documents", headers={"X-API-Key": actor_a})
    assert list_a.status_code == 200
    doc_ids_a = {item["document_id"] for item in list_a.json()["documents"]}
    assert uploaded_a["document_id"] in doc_ids_a
    assert uploaded_b["document_id"] not in doc_ids_a


def test_document_qa_returns_grounded_citations() -> None:
    actor = f"doc_qa_actor_{uuid4().hex}"
    uploaded = _upload_document(
        actor,
        "measure_definition.md",
        "# Measure Definition\nDenominator exclusions apply to patients who transferred out before the measurement period ended.",
    )

    response = client.post(
        "/documents/ask",
        json={"question": "What do the trusted documents say about denominator exclusions?"},
        headers={"X-API-Key": actor},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["citations"]
    assert payload["citations"][0]["document_id"] == uploaded["document_id"]
    assert "Denominator exclusions" in payload["answer"]


def test_pdf_documents_are_ingested_when_text_is_extractable(monkeypatch) -> None:
    actor = f"doc_pdf_actor_{uuid4().hex}"

    class _FakePdfPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakePdfReader:
        def __init__(self, _stream: io.BytesIO) -> None:
            self.pages = [
                _FakePdfPage("Policy update: denominator exclusions require clinical lead approval."),
                _FakePdfPage("Escalate unresolved measure disputes within five business days."),
            ]

    monkeypatch.setattr(main, "PdfReader", _FakePdfReader)

    uploaded = _upload_document(actor, "policy.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")
    assert uploaded["file_type"] == "pdf"

    response = client.post(
        "/documents/ask",
        json={"question": "What do the trusted documents say about denominator exclusions?"},
        headers={"X-API-Key": actor},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["citations"]
    assert payload["citations"][0]["document_id"] == uploaded["document_id"]
    assert "denominator exclusions" in payload["answer"].lower()


def test_viewer_cannot_upload_documents() -> None:
    response = client.post(
        "/documents",
        files={"file": ("viewer.txt", io.BytesIO(b"Viewer should not upload this"), "text/plain")},
        headers={"X-API-Key": "viewer_user", "X-User-Role": "viewer"},
    )
    assert response.status_code == 403
    assert "upload trusted documents" in response.json()["detail"]
