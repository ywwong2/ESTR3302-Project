from backend.embeddings import (
    MODEL_AUDIO,
    MODEL_IMAGE_VIDEO,
    MODEL_TEXT,
    mean_pool,
    max_pool,
    _model_name_for,
)


def test_mean_pool() -> None:
    vectors = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    result = mean_pool(vectors)
    assert result == [2.0, 3.0, 4.0]


def test_max_pool() -> None:
    vectors = [[1.0, 4.0, 3.0], [3.0, 2.0, 5.0]]
    result = max_pool(vectors)
    assert result == [3.0, 4.0, 5.0]


def test_model_name_mapping() -> None:
    assert _model_name_for("image") == MODEL_IMAGE_VIDEO
    assert _model_name_for("video") == MODEL_IMAGE_VIDEO
    assert _model_name_for("audio") == MODEL_AUDIO
    assert _model_name_for("text") == MODEL_TEXT
