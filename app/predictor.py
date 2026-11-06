"""Pembatas antara website dan model.

Seluruh aplikasi memanggil satu fungsi di sini, yaitu prediksi(). Selama model asli
belum ada di folder artifacts/, fungsi itu mengembalikan angka bohongan supaya
tampilan tetap bisa dibangun dan diuji. Begitu berkas aslinya masuk, fungsi ini
memakainya sendiri tanpa ada kode lain yang perlu diubah.

Jangan menulis encoding atau penskalaan di berkas ini. Semua itu tugas
preprocessor.joblib punya Bagas. Kalau ada dua tempat yang melakukan encoding,
cepat atau lambat keduanya akan berbeda, dan website memberi jawaban lain dari
notebook.

PENTING soal versi: preprocessor.joblib dibuat dengan scikit-learn 1.6.1. Pada
scikit-learn 1.9 berkas itu gagal dimuat dengan AttributeError pada
_RemainderColsList. Versi di requirements.txt sudah dikunci, jangan dinaikkan
tanpa membuat ulang preprocessor-nya.
"""

import json
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"

# Urutan kolom mentah, persis seperti CSV Kaggle. Jangan diubah sendirian:
# preprocessor punya Bagas mengharapkan nama dan urutan ini.
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
    """True kalau model asli dan preprocessor sudah ada di artifacts/."""
    return (ART / "model_final.keras").exists() and (ART / "preprocessor.joblib").exists()


def ambang() -> float:
    """Ambang keputusan dari bagian evaluasi.

    Nilai 0.5 di bawah bukan pilihan desain, cuma penahan sementara selama
    threshold.json belum terbit. Angka sebenarnya dicari dari data validasi.
    """
    f = ART / "threshold.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        # Berkas dari src/threshold.py memakai kunci "ambang".
        return float(d.get("ambang", d.get("threshold", 0.5)))
    return 0.5


def _muat():
    """Muat preprocessor dan model sekali saja, lalu simpan di memori."""
    if not _cache:
        import joblib
        from tensorflow import keras

        _cache["pre"] = joblib.load(ART / "preprocessor.joblib")
        _cache["model"] = keras.models.load_model(ART / "model_final.keras")
    return _cache["pre"], _cache["model"]


def prediksi(pasien: dict) -> float:
    """Kembalikan peluang stroke, 0 sampai 1."""
    if not siap():
        # STUB sementara. Angka ini TIDAK berarti apa-apa secara medis maupun
        # statistik. Gunanya cuma supaya tampilan bisa dibangun dan diuji
        # sebelum model asli selesai dilatih.
        skor = (
            (pasien["age"] / 100) * 0.5
            + pasien["hypertension"] * 0.2
            + pasien["heart_disease"] * 0.2
            + (pasien["avg_glucose_level"] / 300) * 0.1
        )
        return round(min(max(skor, 0.0), 1.0), 4)

    pre, model = _muat()
    X = pre.transform(pd.DataFrame([pasien], columns=KOLOM))
    return float(model.predict(X, verbose=0).ravel()[0])


if __name__ == "__main__":
    # Uji cepat: python app/predictor.py
    # Nanti waktu model asli sudah terpasang, ganti contoh ini dengan satu baris
    # dari data uji, lalu cocokkan hasilnya dengan notebook Sherly.
    contoh = {
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }

    p = prediksi(contoh)
    print("model asli terpasang :", siap())
    print("ambang dipakai       :", ambang())
    print("peluang stroke       :", f"{p:.4f}")
    print("masuk kelompok risiko:", p >= ambang())

    assert 0.0 <= p <= 1.0, "peluang harus antara 0 dan 1"
    assert set(contoh) == set(KOLOM), "kunci pasien harus sama persis dengan KOLOM"
    print("\nOK")
