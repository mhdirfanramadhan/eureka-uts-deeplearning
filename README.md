# Prediksi Risiko Stroke pada Data Kesehatan Tidak Seimbang

Tim Eureka — Pembelajaran Mendalam, UTS Semester Ganjil TA 2026/2027.

| Nama | NIM |
|---|---|
| M. Irfan Ramadhan Pasaribu | 231112341 |
| Bagas Arka Dinanta | 231111941 |
| John Alpha Varensius Purba | 231110795 |
| Sherly Angelina Zalukhu | 231112721 |

Dataset: [Stroke Prediction Dataset (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Struktur

```
app/             aplikasi web
src/             praproses, model, pelatihan, evaluasi
notebooks/       notebook EDA dan pelatihan
data/processed/  train.npz, val.npz, test.npz
artifacts/       preprocessor, bobot model, threshold, metrics
figures/         gambar hasil EDA dan evaluasi
```

## Perintah lain

```bash
python src/data.py data/raw/healthcare-dataset-stroke-data.csv
python src/train.py
python src/ekspor_bobot.py
python src/ekspor_praproses.py
python src/verifikasi.py
```

Perintah di atas memerlukan `requirements-dev.txt`. Aplikasi web sendiri hanya
memakai paket pada `requirements.txt`.
