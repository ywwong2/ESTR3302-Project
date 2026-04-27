from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.db import init_db
from backend.main import app

init_db()
client = TestClient(app)


def _png_bytes() -> BytesIO:
    buf = BytesIO()
    Image.new("RGB", (24, 24), color=(200, 40, 40)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_upload_image_file() -> None:
    file_bytes = _png_bytes()
    response = client.post(
        "/media/upload",
        files={"file": ("demo.png", file_bytes, "image/png")},
        data={"title": "Demo Image"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item"]["media_type"] == "image"
    assert payload["item"]["status"] == "UPLOADED"


def test_reject_invalid_content_type() -> None:
    file_bytes = BytesIO(b"hello world")
    response = client.post(
        "/media/upload",
        files={"file": ("bad.txt", file_bytes, "text/plain")},
        data={"title": "Bad"},
    )

    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]
