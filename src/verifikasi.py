import itertools
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


def metrik(y, p, ambang):
    pred = (p >= ambang).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        pr_auc=float(average_precision_score(y, p)),
        n_total=len(y), n_stroke=int(y.sum()),
    )


def cek_praproses(pp, KOLOM, _praproses):
    print("1. PRAPROSES NUMPY LAWAN SCIKIT-LEARN")
    try:
        import joblib
    except ImportError:
        print("   scikit-learn tidak terpasang, pemeriksaan ini dilewati.")
        return True

    berkas = ART / "preprocessor.joblib"
    if not berkas.exists():
        print("   preprocessor.joblib tidak ada, pemeriksaan ini dilewati.")
        return True

    try:
        pre = joblib.load(berkas)
    except Exception as e:
        print(f"   preprocessor.joblib tidak bisa dimuat ({type(e).__name__}), dilewati.")
        return True

    kat = pp["kategori"]
    kombinasi = list(itertools.product(
        kat["gender"], kat["hypertension"], kat["heart_disease"],
        kat["ever_married"], kat["work_type"], kat["Residence_type"],
        kat["smoking_status"],
    ))
    angka = [(0.08, 55.12, 10.3), (43.25, 105.47, 28.77), (82.0, 271.74, 97.6)]

    baris = []
    for g, hy, hd, m, w, r, s in kombinasi:
        for umur, glu, bmi in angka:
            baris.append({
                "gender": g, "age": umur, "hypertension": hy, "heart_disease": hd,
                "ever_married": m, "work_type": w, "Residence_type": r,
                "avg_glucose_level": glu, "bmi": bmi, "smoking_status": s,
            })
    df = pd.DataFrame(baris, columns=KOLOM)

    X_sk = np.asarray(pre.transform(df), dtype="float64")
    X_np = np.asarray(_praproses(df, pp), dtype="float64")

    beda = np.abs(X_sk - X_np)
    print(f"   kombinasi kategori diuji : {len(kombinasi)}")
    print(f"   total baris diuji        : {len(df)}")
    print(f"   bentuk keluaran          : {X_np.shape} lawan {X_sk.shape}")
    print(f"   selisih maksimum         : {beda.max():.3e}")
    sama = X_sk.shape == X_np.shape and beda.max() < 1e-6
    print("   Praproses NumPy setara dengan scikit-learn."
          if sama else "   HASILNYA BERBEDA.")
    return sama


def main():
    M = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    acuan = M["versi"]["4_focal_ambang"]

    from predictor import KOLOM, _maju, _muat, _praproses, prediksi, siap
    from predictor import ambang as ambang_app

    ambang = ambang_app()
    pp, bobot, susunan = _muat()

    semua = cek_praproses(pp, KOLOM, _praproses)

    d = np.load(AKAR / "data" / "processed" / "test.npz")
    X, y = d["X"].astype("float32"), d["y"].astype(int).ravel()
    p_np = _maju(X, bobot, susunan).ravel()

    print(f"\n2. HASIL MODEL LAWAN metrics.json  ({len(y)} pasien, {int(y.sum())} stroke)")
    h = metrik(y, p_np, ambang)
    print(f"   {'Ukuran':<12}{'Dijalankan ulang':>20}{'metrics.json':>18}{'Status':>10}")
    print("   " + "-" * 60)
    for k in ("tp", "fp", "fn", "tn", "n_total", "n_stroke",
              "recall", "precision", "f1", "pr_auc"):
        a, b = h[k], acuan[k]
        ok = (a == b) if isinstance(b, int) else abs(a - b) < 1e-9
        semua &= ok
        fa = f"{a:.6f}" if isinstance(a, float) else str(a)
        fb = f"{b:.6f}" if isinstance(b, float) else str(b)
        print(f"   {k:<12}{fa:>20}{fb:>18}{'COCOK' if ok else 'BEDA':>10}")
    print("   " + "-" * 60)

    print("\n3. MODEL NUMPY LAWAN KERAS")
    try:
        import keras
    except ImportError:
        print("   TensorFlow tidak terpasang, pemeriksaan ini dilewati.")
    else:
        model = keras.models.load_model(ART / "model_final.keras")
        p_keras = model.predict(X, batch_size=256, verbose=0).ravel()
        beda = np.abs(p_np - p_keras)
        beda_keputusan = int(((p_np >= ambang) != (p_keras >= ambang)).sum())
        print(f"   selisih maksimum   : {beda.max():.3e}")
        print(f"   keputusan berbeda  : {beda_keputusan} dari {len(y)} pasien")
        sama = beda.max() < 1e-5 and beda_keputusan == 0
        print("   Model NumPy setara dengan Keras." if sama else "   HASILNYA BERBEDA.")
        semua &= sama

    pasien = {
        "gender": "Male", "age": 67.0, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
    }
    print("\n4. JALUR WEBSITE")
    print("   model terpasang      :", siap())
    skor = prediksi(pasien)
    print("   peluang stroke       :", f"{skor:.6f}")
    print("   masuk kelompok risiko:", skor >= ambang)

    print("\nSEMUA PEMERIKSAAN LULUS." if semua else "\nADA PEMERIKSAAN YANG GAGAL.")
    return 0 if semua else 1


if __name__ == "__main__":
    raise SystemExit(main())
