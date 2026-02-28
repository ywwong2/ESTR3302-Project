from io import BytesIO

from fastapi.testclient import TestClient

from backend.db import init_db
from backend.main import app

init_db()
client = TestClient(app)


def test_upload_text_file() -> None:
    file_bytes = BytesIO(b"hello multimedia")
    response = client.post(
        "/media/upload",
        files={"file": ("demo.txt", file_bytes, "text/plain")},
        data={"title": "Demo Text"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item"]["status"] == "UPLOADED"


def test_reject_invalid_content_type() -> None:
    file_bytes = BytesIO(b"{}")
    response = client.post(
        "/media/upload",
        files={"file": ("bad.json", file_bytes, "application/json")},
        data={"title": "Bad"},
    )

    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]
