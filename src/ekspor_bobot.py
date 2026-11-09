import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

AKAR = Path(__file__).resolve().parents[1]
ART = AKAR / "artifacts"
KELUARAN = ART / "bobot_model.npz"


def main():
    import keras

    model = keras.models.load_model(ART / "model_final.keras")
    bobot = {}
    lapisan = []

    for lap in model.layers:
        jenis = lap.__class__.__name__
        w = lap.get_weights()
        if jenis == "Dense":
            bobot[f"{lap.name}__kernel"] = w[0]
            if len(w) > 1:
                bobot[f"{lap.name}__bias"] = w[1]
            lapisan.append({"nama": lap.name, "jenis": "Dense",
                            "pakai_bias": len(w) > 1,
                            "aktivasi": keras.activations.serialize(lap.activation)})
        elif jenis == "BatchNormalization":
            gamma, beta, rata, varian = w
            for k, v in (("gamma", gamma), ("beta", beta),
                         ("rata", rata), ("varian", varian)):
                bobot[f"{lap.name}__{k}"] = v
            lapisan.append({"nama": lap.name, "jenis": "BatchNormalization",
                            "epsilon": float(lap.epsilon)})
        elif jenis == "Activation":
            lapisan.append({"nama": lap.name, "jenis": "Activation",
                            "aktivasi": keras.activations.serialize(lap.activation)})
        elif jenis in ("Dropout", "InputLayer"):
            continue
        else:
            raise SystemExit(f"lapisan belum didukung: {jenis} ({lap.name})")

    n_fitur = int(model.inputs[0].shape[1])
    np.savez(KELUARAN,
             susunan=json.dumps({"n_fitur": n_fitur, "lapisan": lapisan}),
             **bobot)

    total = sum(int(np.prod(v.shape)) for v in bobot.values())
    print(f"lapisan diekspor : {len(lapisan)}")
    print(f"jumlah parameter : {total}  (model.count_params(): {model.count_params()})")
    print(f"ukuran berkas    : {KELUARAN.stat().st_size / 1024:.1f} KB")
    print(f"tersimpan        : {KELUARAN}")

    assert total == model.count_params()
    print("\nOK")


if __name__ == "__main__":
    main()
