from backend.embeddings import (
    MODEL_IMAGE_VIDEO,
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


def test_model_name_mapping_rejects_non_image() -> None:
    try:
        _model_name_for("audio")
        assert False, "Expected ValueError for non-image media type"
    except ValueError as exc:
        assert "Unsupported media type" in str(exc)
