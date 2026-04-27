from pathlib import Path

from PIL import Image

from backend.preprocess import (
    IMAGE_TARGET_SIZE,
    detect_media_type,
    preprocess_image,
    run_preprocessing,
)


def test_detect_media_type_image_only() -> None:
    assert detect_media_type("image/png") == "image"


def test_detect_media_type_rejects_non_image() -> None:
    try:
        detect_media_type("text/plain")
        assert False, "Expected ValueError for non-image content"
    except ValueError as exc:
        assert "Unsupported content type" in str(exc)


def test_preprocess_image_resizes_and_writes_output(tmp_path: Path) -> None:
    src = tmp_path / "source.png"
    Image.new("RGB", (640, 360), color=(120, 20, 30)).save(src)

    out_dir = tmp_path / "out"
    result = preprocess_image(src, out_dir)

    assert result["modality"] == "image"
    assert result["target_size"] == list(IMAGE_TARGET_SIZE)
    assert result["original_size"] == [640, 360]
    assert Path(result["processed_path"]).exists()


def test_run_preprocessing_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "source.png"
    Image.new("RGB", (320, 320), color=(0, 100, 180)).save(src)

    out_dir = tmp_path / "preprocessed"
    result = run_preprocessing("image", src, out_dir)

    assert result["modality"] == "image"
    assert Path(result["manifest_path"]).exists()
