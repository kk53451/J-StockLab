"""Optional TensorFlow models; importing the baseline CLI does not require TF."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data import STOCKS, Windows


@keras.utils.register_keras_serializable(package="JStockLab")
class PositionEncoding(keras.layers.Layer):
    def __init__(self, length: int, width: int, **kwargs):
        super().__init__(**kwargs)
        self.length, self.width = length, width
        position = np.arange(length)[:, None]
        angle = position / np.power(10000, (2 * (np.arange(width) // 2)) / width)
        values = np.where(np.arange(width) % 2 == 0, np.sin(angle), np.cos(angle))
        self.encoding = values[None].astype(np.float32)

    def call(self, inputs):
        return inputs + tf.cast(self.encoding, inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "length": self.length, "width": self.width}


def build_model(kind: str, lookback: int, feature_count: int, horizon: int,
                units: int = 64, position_encoding: bool = True):
    if units < 4 or units % 4:
        raise ValueError("units must be a positive multiple of 4")
    inputs, encoded = {}, []
    sizes = {"stock_input": len(STOCKS)}
    if feature_count > len(STOCKS):
        sizes["econ_input"] = feature_count - len(STOCKS)
    for name, size in sizes.items():
        inputs[name] = keras.Input((lookback, size), name=name)
        x = inputs[name]
        if kind == "lstm":
            x = keras.layers.LSTM(units, return_sequences=True)(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.LSTM(units)(x)
            x = keras.layers.Dropout(0.2)(x)
        elif kind == "transformer":
            x = keras.layers.Dense(units)(x)
            if position_encoding:
                x = PositionEncoding(lookback, units)(x)
            for _ in range(2):
                attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=units // 4)(x, x)
                x = keras.layers.LayerNormalization()(x + keras.layers.Dropout(0.1)(attention))
                feedforward = keras.layers.Dense(units * 2, activation="relu")(x)
                feedforward = keras.layers.Dense(units)(feedforward)
                x = keras.layers.LayerNormalization()(x + keras.layers.Dropout(0.1)(feedforward))
            x = keras.layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(f"Unknown neural model: {kind}")
        encoded.append(keras.layers.Dense(64, activation="relu")(x))
    merged = keras.layers.Concatenate()(encoded) if len(encoded) > 1 else encoded[0]
    merged = keras.layers.Dense(128, activation="relu")(merged)
    merged = keras.layers.Dropout(0.2)(merged)
    outputs = keras.layers.Dense(horizon * len(STOCKS))(merged)
    return keras.Model(inputs, outputs)


def stream_inputs(x):
    result = {"stock_input": x[:, :, :len(STOCKS)]}
    if x.shape[-1] > len(STOCKS):
        result["econ_input"] = x[:, :, len(STOCKS):]
    return result


def batches(windows: Windows, batch_size: int, with_targets: bool):
    inputs = stream_inputs(windows.x)
    elements = (inputs, windows.y_scaled.reshape(len(windows.y), -1)) if with_targets else inputs
    dataset = tf.data.Dataset.from_tensor_slices(elements).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)


def train_model(kind, train, validation, settings):
    keras.utils.set_random_seed(settings["seed"])
    tf.config.experimental.enable_op_determinism()
    model = build_model(kind, train.x.shape[1], train.x.shape[2], train.y.shape[1],
                        settings["units"], settings["position_encoding"])
    model.compile(optimizer=keras.optimizers.Adam(settings["learning_rate"]), loss="mse")
    history = model.fit(
        batches(train, settings["batch_size"], True),
        validation_data=batches(validation, settings["batch_size"], True),
        epochs=settings["epochs"], verbose=2, shuffle=False,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=settings["patience"], restore_best_weights=True,
        )],
    )
    return model, history.history


def predict_model(model, windows, batch_size):
    return model.predict(batches(windows, batch_size, False), verbose=0)
