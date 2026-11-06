"""Pemeriksaan konsistensi akhir.

Menjalankan ulang model final pada data uji, lalu membandingkan hasilnya dengan
angka yang tercatat di artifacts/metrics.json. Kalau ada yang tidak cocok,
berarti laporan dan kode sudah berbeda dan salah satunya harus diperbaiki.

Jalankan dari akar repo:
    python src/verifikasi.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

AKAR = Path(__file__).resolve().parents[1]
ART = AKAR / "artifacts"
sys.path.insert(0, str(AKAR / "app"))


def main():
    M = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    T = json.loads((ART / "threshold.json").read_text(encoding="utf-8"))
    acuan = M["versi"]["4_focal_ambang"]
    ambang = float(T["ambang"])

    import keras

    model = keras.models.load_model(ART / "model_final.keras")
    d = np.load(AKAR / "data" / "processed" / "test.npz")
    X, y = d["X"].astype("float32"), d["y"].astype(int).ravel()

    p = model.predict(X, batch_size=256, verbose=0).ravel()
    pred = (p >= ambang).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pr_auc = float(average_precision_score(y, p))

    hitung = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
                  f1=f1, pr_auc=pr_auc, n_total=len(y), n_stroke=int(y.sum()))

    print(f"model   : {ART / 'model_final.keras'}")
    print(f"data uji: {len(y)} pasien, {int(y.sum())} kasus stroke")
    print(f"ambang  : {ambang:.6f}\n")
    print(f"{'Ukuran':<12}{'Dijalankan ulang':>20}{'metrics.json':>18}{'Status':>10}")
    print("-" * 60)

    semua_cocok = True
    for k in ("tp", "fp", "fn", "tn", "n_total", "n_stroke",
              "recall", "precision", "f1", "pr_auc"):
        a, b = hitung[k], acuan[k]
        cocok = (a == b) if isinstance(b, int) else abs(a - b) < 1e-9
        semua_cocok &= cocok
        fa = f"{a:.6f}" if isinstance(a, float) else str(a)
        fb = f"{b:.6f}" if isinstance(b, float) else str(b)
        print(f"{k:<12}{fa:>20}{fb:>18}{'COCOK' if cocok else 'BEDA':>10}")

    print("-" * 60)
    print("Seluruh angka cocok dengan metrics.json."
          if semua_cocok else "ADA YANG TIDAK COCOK.")

    # --- jalur website: dict pasien mentah, bukan matriks yang sudah diproses ---
    from predictor import ambang as ambang_app
    from predictor import prediksi, siap

    pasien = {
        "gender": "Male", "age": 67.0, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
    }
    print("\n--- jalur website (app/predictor.py) ---")
    print("model asli terpasang :", siap())
    print("ambang dipakai       :", f"{ambang_app():.6f}")
    skor = prediksi(pasien)
    print("peluang stroke       :", f"{skor:.6f}")
    print("masuk kelompok risiko:", skor >= ambang_app())

    return 0 if semua_cocok else 1


if __name__ == "__main__":
    raise SystemExit(main())
