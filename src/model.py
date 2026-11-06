"""Arsitektur MLP.

Penulis: John Alpha Varensius Purba (231110795).
Dikonversi dari notebooks/02_model_john.ipynb.

Dua hidden layer. Tiap hidden layer disusun Dense (tanpa bias) -> BatchNorm ->
ReLU -> Dropout, dengan L2 pada bobot Dense. Keluaran satu neuron sigmoid,
tanpa BatchNorm maupun Dropout.

Dense dibuat tanpa bias karena parameter geser beta pada BatchNormalization
sudah berperan sebagai bias, sehingga bias pada Dense jadi berlebih.
"""

import keras
from keras import layers

CONFIG = dict(units=[32, 16], dropout=0.2, l2=1e-4, lr=1e-3, batch=64,
              epochs=200, patience=15, gamma=2.0, alpha=0.25)


def bangun_model(n_fitur, units=(32, 16), dropout=0.2, l2=1e-4):
    reg = keras.regularizers.L2(l2)
    blok = [keras.Input(shape=(n_fitur,))]
    for u in units:
        blok += [
            layers.Dense(u, use_bias=False, kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout),
        ]
    blok.append(layers.Dense(1, activation="sigmoid"))
    return keras.Sequential(blok, name="mlp_stroke")


def rincian_parameter(n_fitur, units=(32, 16)):
    """Hitung parameter dengan rumus, untuk dicocokkan dengan count_params()."""
    u1, u2 = units
    return {
        "Dense 1": n_fitur * u1,      # tanpa bias
        "BatchNorm 1": 4 * u1,        # gamma, beta, rata-rata, varians
        "Dense 2": u1 * u2,
        "BatchNorm 2": 4 * u2,
        "Output": u2 + 1,             # dengan bias
    }


if __name__ == "__main__":
    import numpy as np

    N_FITUR = 15
    m = bangun_model(N_FITUR, CONFIG["units"], CONFIG["dropout"], CONFIG["l2"])
    m.summary()

    rincian = rincian_parameter(N_FITUR, CONFIG["units"])
    for nama, n in rincian.items():
        print(f"{nama:12s} {n:6d}")
    manual = sum(rincian.values())
    trainable = int(sum(np.prod(w.shape) for w in m.trainable_weights))
    print(f"Total manual : {manual}")
    print(f"Total Keras  : {m.count_params()} (dilatih {trainable})")
    assert manual == m.count_params(), "hitungan manual beda dengan Keras"
    print("COCOK")
