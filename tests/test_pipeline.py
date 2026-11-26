import numpy as np
from PIL import Image

from src.cnn_numpy import TinyCNN
from src.data import load_real_digits
from src.inference import predict_image, preprocess_uploaded_image


def test_dataset_is_stratified_and_normalized():
    dataset = load_real_digits(seed=42)
    assert dataset.X_train.shape == (1257, 8, 8)
    assert dataset.X_val.shape == (270, 8, 8)
    assert dataset.X_test.shape == (270, 8, 8)
    assert 0.0 <= float(dataset.X_train.min())
    assert float(dataset.X_train.max()) <= 1.0
    assert set(np.unique(dataset.y_train)) == set(range(10))


def test_model_round_trip(tmp_path):
    model = TinyCNN(conv_filters=4, hidden_dim=8, seed=7)
    before = model.predict_proba(np.zeros((2, 8, 8), dtype=np.float32))
    path = tmp_path / "model.npz"
    model.save(path)
    restored = TinyCNN.load(path)
    after = restored.predict_proba(np.zeros((2, 8, 8), dtype=np.float32))
    np.testing.assert_allclose(before, after, atol=1e-7)


def test_preprocessing_and_inference_shapes():
    source = Image.new("L", (32, 32), color=255)
    processed = preprocess_uploaded_image(source)
    assert processed.shape == (8, 8)
    assert np.isfinite(processed).all()
    model = TinyCNN(conv_filters=4, hidden_dim=8, seed=9)
    result = predict_image(model, source)
    assert 0 <= result["prediction"] <= 9
    assert 0.0 <= result["confidence"] <= 1.0
