import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"

KOLOM = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

_cache = {}


def siap() -> bool:
    return (ART / "bobot_model.npz").exists() and (ART / "praproses.json").exists()


def ambang() -> float:
    f = ART / "threshold.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return float(d.get("ambang", d.get("threshold", 0.5)))
    return 0.5


def _muat():
    if not _cache:
        d = np.load(ART / "bobot_model.npz")
        _cache["bobot"] = {k: d[k] for k in d.files if k != "susunan"}
        _cache["susunan"] = json.loads(str(d["susunan"]))
        _cache["pp"] = json.loads((ART / "praproses.json").read_text(encoding="utf-8"))
    return _cache["pp"], _cache["bobot"], _cache["susunan"]


def _praproses(df, pp):
    num = df[pp["num_cols"]].to_numpy(dtype="float64")
    bagian = [(num - np.asarray(pp["mean"])) / np.asarray(pp["scale"])]

    for kol in pp["cat_cols"]:
        kat = pp["kategori"][kol]
        oh = np.zeros((len(df), len(kat)), dtype="float64")
        for i, nilai in enumerate(df[kol].tolist()):
            cocok = [j for j, c in enumerate(kat) if c == nilai]
            if not cocok:
                raise ValueError(f"nilai tidak dikenal untuk {kol}: {nilai!r}")
            oh[i, cocok[0]] = 1.0
        bagian.append(oh[:, 1:] if pp["buang_kategori_pertama"] else oh)

    return np.hstack(bagian).astype("float32")


def _aktivasi(x, nama):
    if nama in (None, "linear"):
        return x
    if nama == "relu":
        return np.maximum(x, 0.0)
    if nama == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"aktivasi belum didukung: {nama}")


def _maju(X, bobot, susunan):
    h = np.asarray(X, dtype="float32")

    for lap in susunan["lapisan"]:
        nama, jenis = lap["nama"], lap["jenis"]

        if jenis == "Dense":
            h = h @ bobot[f"{nama}__kernel"]
            if lap["pakai_bias"]:
                h = h + bobot[f"{nama}__bias"]
            h = _aktivasi(h, lap["aktivasi"])

        elif jenis == "BatchNormalization":
            g = bobot[f"{nama}__gamma"]
            b = bobot[f"{nama}__beta"]
            m = bobot[f"{nama}__rata"]
            v = bobot[f"{nama}__varian"]
            h = g * (h - m) / np.sqrt(v + lap["epsilon"]) + b

        elif jenis == "Activation":
            h = _aktivasi(h, lap["aktivasi"])

        else:
            raise ValueError(f"lapisan belum didukung: {jenis}")

    return h


def prediksi_banyak(df: pd.DataFrame) -> np.ndarray:
    pp, bobot, susunan = _muat()
    X = _praproses(df[KOLOM], pp)
    return _maju(X, bobot, susunan).ravel()


def prediksi(pasien: dict) -> float:
    if not siap():
        skor = (
            (pasien["age"] / 100) * 0.5
            + pasien["hypertension"] * 0.2
            + pasien["heart_disease"] * 0.2
            + (pasien["avg_glucose_level"] / 300) * 0.1
        )
        return round(min(max(skor, 0.0), 1.0), 4)

    return float(prediksi_banyak(pd.DataFrame([pasien], columns=KOLOM))[0])


if __name__ == "__main__":
    contoh = {
        "gender": "Male", "age": 67.0, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
    }
    p = prediksi(contoh)
    print("model terpasang      :", siap())
    print("ambang dipakai       :", f"{ambang():.6f}")
    print("peluang stroke       :", f"{p:.6f}")
    print("masuk kelompok risiko:", p >= ambang())
