from pathlib import Path

from backend.preprocess import (
    TEXT_MAX_CHARS,
    uniform_audio_chunks,
    uniform_timestamps,
    preprocess_text,
)


def test_uniform_timestamps_deterministic() -> None:
    result = uniform_timestamps(20.0, 10)
    assert len(result) == 10
    assert result[0] == 1.0
    assert result[-1] == 19.0


def test_uniform_audio_chunks_count() -> None:
    chunks = uniform_audio_chunks(30.0, 10)
    assert len(chunks) == 10
    assert chunks[0] == {"start": 0.0, "end": 3.0}
    assert chunks[-1] == {"start": 27.0, "end": 30.0}


def test_text_trim(tmp_path: Path) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("A" * (TEXT_MAX_CHARS + 10), encoding="utf-8")

    result = preprocess_text(text_file)
    assert result["final_length"] == TEXT_MAX_CHARS
