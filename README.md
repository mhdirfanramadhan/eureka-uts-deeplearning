# Prediksi Risiko Stroke pada Data Kesehatan Tidak Seimbang

Proyek UTS Pembelajaran Mendalam, Tim Eureka.
Model: MLP (Deep Neural Network) dengan 2 hidden layer, BatchNormalization, Dropout, dan L2.

## Anggota

| Nama | NIM | Bagian |
|---|---|---|
| M. Irfan Ramadhan Pasaribu | 231112341 | Website, integrasi, ketua |
| Bagas Arka Dinanta | 231111941 | Data dan EDA |
| John Alpha Varensius Purba | 231110795 | Arsitektur dan pelatihan |
| Sherly Angelina Zalukhu | 231112721 | Evaluasi dan ambang keputusan |

## Dataset

[Stroke Prediction Dataset (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Sekitar 5.110 baris dengan 10 kolom fitur. Hanya sekitar 4,9 persen baris yang
berlabel stroke, jadi datanya sangat tidak seimbang. Angka pastinya dikonfirmasi
Bagas saat EDA, dan angka hasil konfirmasi itu yang dipakai di laporan.

## Tiga aturan

1. **Akurasi bukan patokan.** Menjawab "tidak stroke" untuk semua orang sudah
   memberi akurasi sekitar 95 persen tanpa belajar apa pun. Yang dipakai:
   **recall** kelas stroke dan **PR-AUC**.
2. **Semua angka laporan berasal dari `artifacts/metrics.json`.** Angka yang
   tidak ada di situ tidak boleh masuk laporan.
3. **Keputusan bersama tidak diubah sendirian.** Daftarnya ada di bawah.

## Keputusan yang dikunci

| Apa | Nilai |
|---|---|
| Random seed | 42 |
| Pembagian data | 70 / 15 / 15, stratified |
| Metrik penentu | PR-AUC, lalu recall |
| Scaler | di-fit pada data latih saja |
| Ambang keputusan | dicari dari data validasi |
| Data test | dibuka sekali, di akhir |

## Struktur folder

```
app/             website Streamlit              (Irfan)
src/             kode hasil konsolidasi .py     (dari Bagas, John, Sherly)
notebooks/       notebook asli, disimpan apa adanya
data/raw/        CSV mentah, tidak di-commit
data/processed/  train/val/test.npz             (Bagas)
artifacts/       preprocessor, model, metrics   (Bagas, John, Sherly)
figures/         grafik untuk laporan           (Sherly)
pengerjaan-tim/  catatan kontribusi dan bukti kerja tim
```

Berkas laporan tidak disimpan di repositori ini. Laporan dikumpulkan terpisah
sebagai satu berkas PDF sesuai ketentuan tugas.

Notebook asli dari masing-masing anggota disimpan apa adanya di `notebooks/`,
berdampingan dengan versi `.py` hasil konsolidasi di `src/`. Keduanya disimpan, bukan
salah satu: output sel di notebook adalah hasil eksekusi di mesin penulisnya, dan itu
jejak kontribusi yang tidak bisa dihasilkan ulang oleh orang lain.

Folder `artifacts/` adalah titik temu semua orang:

| Berkas | Dibuat oleh | Dipakai oleh |
|---|---|---|
| `preprocessor.joblib` | Bagas | John, Sherly, website |
| `model_dasar.keras` | John | Sherly |
| `model_weighted.keras` | John | Sherly |
| `model_focal.keras` | John | Sherly |
| `model_final.keras` | Sherly | website |
| `threshold.json` | Sherly | website, laporan |
| `metrics.json` | Sherly | laporan semua orang |

## Menjalankan website

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Selama `artifacts/` masih kosong, website jalan memakai angka contoh dan
menampilkan peringatan. Begitu `preprocessor.joblib` dan `model_final.keras`
masuk, website memakainya sendiri tanpa perubahan kode.

Uji cepat pembatas model, tanpa Streamlit:

```bash
python app/predictor.py
```

## Catatan penting soal versi

Model Keras dan preprocessor scikit-learn bisa gagal dimuat di versi yang
berbeda dari waktu menyimpannya. Versi di `requirements.txt` harus sama dengan
yang dipakai John waktu melatih. Jalankan ini di Colab dan kirim hasilnya:

```python
import tensorflow, sklearn, numpy, pandas
print(tensorflow.__version__, sklearn.__version__, numpy.__version__, pandas.__version__)
```

## Hasil

Empat versi dievaluasi pada 767 pasien uji yang memuat 38 kasus stroke. Ambang
dicari dari data validasi sebelum data uji dibuka satu kali.

| Versi | Ambang | Stroke terdeteksi | Recall | Precision | PR-AUC | Salah alarm |
|---|---|---|---|---|---|---|
| Dasar | 0,500 | 0 dari 38 | 0,000 | 0,000 | 0,143 | 0 |
| Class weight | 0,500 | 28 dari 38 | 0,737 | 0,110 | 0,165 | 226 |
| Focal loss | 0,500 | 0 dari 38 | 0,000 | 0,000 | 0,161 | 0 |
| **Focal + ambang** | 0,260 | 21 dari 38 | 0,553 | 0,181 | 0,161 | 95 |

Versi final adalah Focal + ambang, dipilih dari F2 tertinggi di data validasi.
Angka di atas berasal dari `artifacts/metrics.json`. Sebagai pembanding,
menebak "tidak stroke" untuk semua pasien memberi akurasi 0,950 dengan nol
pasien stroke terdeteksi.

## Status

- [x] Data bersih, EDA, dan pembagian 70/15/15
- [x] Tiga model dilatih (dasar, class weight, focal loss)
- [x] Evaluasi, pencarian ambang, dan `metrics.json`
- [x] Website dengan form sepuluh kolom, model asli terpasang
- [x] Laporan UTS tersusun (disimpan terpisah, tidak di repositori ini)
- [ ] Deploy ke hosting publik

## Pemeriksaan konsistensi

```bash
python src/verifikasi.py
```

Skrip ini memuat `model_final.keras`, menjalankannya pada seluruh data uji,
lalu membandingkan hasilnya dengan `artifacts/metrics.json`. Kesepuluh ukuran
cocok sampai enam angka di belakang koma:

| Ukuran | Dijalankan ulang | metrics.json |
|---|---|---|
| TP / FP / FN / TN | 21 / 95 / 17 / 634 | 21 / 95 / 17 / 634 |
| Recall | 0,552632 | 0,552632 |
| Precision | 0,181034 | 0,181034 |
| F1 | 0,272727 | 0,272727 |
| PR-AUC | 0,161027 | 0,161027 |

Jalankan skrip ini setiap kali ada artefak yang diganti. Kalau ada yang tidak
cocok, laporan dan kode sudah berbeda dan salah satunya harus diperbaiki.

Tiga model varian milik bagian pelatihan (`model_dasar`, `model_weighted`,
`model_focal`) tidak ikut disimpan di repositori; yang tersimpan hanya
`model_final.keras` yang dipakai website. Ketiganya dapat dibuat ulang dengan
menjalankan `python src/train.py`.
