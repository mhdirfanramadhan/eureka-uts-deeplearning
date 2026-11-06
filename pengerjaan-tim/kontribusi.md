# Catatan Kontribusi

Diisi sambil jalan, jangan ditulis belakangan sekaligus. Tabel pertama disalin langsung
jadi Tabel Pembagian Tugas di Bagian Awal nomor 4 laporan.

## Pembagian tugas

| NIM | Nama | Peran | Rincian pekerjaan |
|---|---|---|---|
| 231111941 | Bagas Arka Dinanta | Data dan EDA | Pembersihan data, penanganan BMI kosong dan kategori Unknown, encoding, standardisasi, pembagian data 70/15/15, EDA. Menulis Bagian A.1a–b, A.2, A.3, A.4. |
| 231110795 | John Alpha Varensius Purba | Arsitektur dan pelatihan | MLP 2 hidden layer dengan BatchNorm, Dropout, L2. Melatih tiga model: dasar, class weight, focal loss. Fitur unik 1. Menulis Bagian A.5a–d dan A.5e bagian 1. |
| 231112721 | Sherly Angelina Zalukhu | Evaluasi dan ambang | Precision, recall, F1, PR-AUC, confusion matrix, kurva loss, analisis kesalahan. Pencarian ambang keputusan sebagai fitur unik 2. Menulis Bagian A.5e bagian 2 dan A.6. |
| 231112341 | M. Irfan Ramadhan Pasaribu | Website dan integrasi | Website prediksi, repo, konsolidasi kode jadi `.py`, pemasangan model, deployment, perakitan laporan. Menulis Cover, Bagian Awal, A.1c, A.7, dan Bagian B. |

## Catatan harian

Satu baris tiap ada serah-terima atau keputusan. Isi tanggalnya yang sebenarnya.

| Tanggal | Siapa | Apa |
|---|---|---|
| | Irfan | Repo dibuat, struktur folder dan kerangka website dengan model contoh |
| | | |
| | | |

## Serah-terima berkas

Dicentang waktu berkasnya benar-benar sudah masuk repo, bukan waktu dijanjikan.

**Bagas**
- [ ] `src/data.py`
- [ ] `notebooks/01_eda.ipynb`
- [ ] `data/processed/train.npz`, `val.npz`, `test.npz`
- [ ] `artifacts/preprocessor.joblib`
- [ ] Gambar EDA ke `figures/`
- [ ] Tulisan A.1a–b, A.2, A.3, A.4

**John**
- [ ] `src/model.py`, `src/losses.py`, `src/train.py`
- [ ] `artifacts/model_dasar.keras`, `model_weighted.keras`, `model_focal.keras`
- [ ] Diagram arsitektur dan keluaran `model.summary()`
- [ ] Versi tensorflow, scikit-learn, numpy, pandas
- [ ] Tulisan A.5a–d dan fitur unik 1

**Sherly**
- [ ] `src/threshold.py`, `src/evaluate.py`
- [ ] `artifacts/model_final.keras`, `threshold.json`, `metrics.json`
- [ ] Grafik ke `figures/`
- [ ] Tulisan fitur unik 2 dan A.6

**Irfan**
- [x] Repo, struktur folder, `.gitignore`, `requirements.txt`
- [x] `app/predictor.py` dan `app/app.py`, jalan dengan model contoh
- [ ] Model asli terpasang dan cocok dengan notebook Sherly
- [ ] Deploy, tautan terbuka tanpa login
- [ ] Laporan dirakit jadi satu PDF
