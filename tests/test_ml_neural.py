import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml_eval.neural import build_model, keras, stream_inputs


@pytest.mark.parametrize("kind,features", [("lstm", 20), ("lstm", 23), ("transformer", 20), ("transformer", 23)])
def test_model_learns_one_batch_and_serializes(tmp_path, kind, features):
    keras.utils.set_random_seed(7)
    model = build_model(kind, 5, features, 2, units=8)
    inputs = stream_inputs(np.random.default_rng(7).normal(size=(3, 5, features)).astype("float32"))
    model.compile(optimizer="adam", loss="mse")
    loss = model.train_on_batch(inputs, np.zeros((3, 40), dtype="float32"))
    assert np.isfinite(loss)
    expected = model(inputs, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected, atol=1e-6)


def test_position_encoding_breaks_shared_time_permutation_invariance():
    x = np.random.default_rng(13).normal(size=(3, 5, 23)).astype("float32")
    for use_position in (False, True):
        keras.utils.set_random_seed(7)
        model = build_model("transformer", 5, 23, 2, units=8, position_encoding=use_position)
        forward = model(stream_inputs(x), training=False).numpy()
        backward = model(stream_inputs(x[:, ::-1]), training=False).numpy()
        if use_position:
            assert np.max(np.abs(forward - backward)) > 1e-4
        else:
            np.testing.assert_allclose(forward, backward, atol=2e-6)
