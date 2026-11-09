import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from losses import focal_loss
from model import CONFIG, bangun_model

SEED = 42
AKAR = Path(__file__).resolve().parents[1]
DIR_PROC = AKAR / "data" / "processed"
DIR_ART = AKAR / "artifacts"


def muat_data():
    d_tr = np.load(DIR_PROC / "train.npz")
    d_va = np.load(DIR_PROC / "val.npz")
    X_tr, y_tr = d_tr["X"].astype("float32"), d_tr["y"].astype("float32")
    X_va, y_va = d_va["X"].astype("float32"), d_va["y"].astype("float32")

    assert X_tr.shape[1] == X_va.shape[1]
    assert set(np.unique(y_tr)) == {0.0, 1.0}
    assert not np.isnan(X_tr).any() and not np.isnan(X_va).any()

    print(f"Latih   : {len(y_tr)} baris, {int(y_tr.sum())} stroke")
    print(f"Validasi: {len(y_va)} baris, {int(y_va.sum())} stroke")
    print(f"Jumlah fitur: {X_tr.shape[1]}")
    return X_tr, y_tr, X_va, y_va


def latih(X_tr, y_tr, X_va, y_va, loss, class_weight=None, verbose=2):
    keras.utils.set_random_seed(SEED)
    model = bangun_model(X_tr.shape[1], CONFIG["units"], CONFIG["dropout"], CONFIG["l2"])
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["lr"]),
        loss=loss,
        metrics=[keras.metrics.AUC(curve="PR", name="auc_pr"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.Precision(name="precision")],
    )
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(len(X_tr), seed=SEED, reshuffle_each_iteration=True)
                .batch(CONFIG["batch"]).prefetch(1))
    es = keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max",
                                       patience=CONFIG["patience"],
                                       restore_best_weights=True)
    h = model.fit(train_ds, validation_data=(X_va, y_va), epochs=CONFIG["epochs"],
                  class_weight=class_weight, callbacks=[es], verbose=verbose)
    return model, {k: [float(x) for x in v] for k, v in h.history.items()}


def main():
    X_tr, y_tr, X_va, y_va = muat_data()
    n0, n1 = np.bincount(y_tr.astype(int))
    DIR_ART.mkdir(parents=True, exist_ok=True)

    varian = {
        "dasar": dict(loss="binary_crossentropy", class_weight=None),
        "weighted": dict(loss="binary_crossentropy",
                         class_weight={0: 1.0, 1: float(n0 / n1)}),
        "focal": dict(loss=focal_loss(CONFIG["gamma"], CONFIG["alpha"]),
                      class_weight=None),
    }

    log = dict(config=CONFIG, seed=SEED, n_fitur=int(X_tr.shape[1]),
               latih=dict(tidak_stroke=int(n0), stroke=int(n1)), model={})

    for nama, v in varian.items():
        print(f"\n=== model_{nama} ===")
        model, hist = latih(X_tr, y_tr, X_va, y_va, v["loss"], v["class_weight"])
        model.save(DIR_ART / f"model_{nama}.keras")
        terbaik = int(np.argmax(hist["val_auc_pr"]))
        log["model"][nama] = dict(
            class_weight=v["class_weight"],
            parameter_total=int(model.count_params()),
            epoch_jalan=len(hist["loss"]),
            epoch_terbaik=terbaik + 1,
            val_auc_pr=hist["val_auc_pr"][terbaik],
            val_recall=hist["val_recall"][terbaik],
            val_precision=hist["val_precision"][terbaik],
            history=hist,
        )

    (DIR_ART / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("\nTersimpan di", DIR_ART)


if __name__ == "__main__":
    main()
