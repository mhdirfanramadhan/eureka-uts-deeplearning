# Pengerjaan Tim

Catatan proses kerja Tim Eureka. Isinya bukti bahwa pekerjaan ini dikerjakan berempat,
bukan oleh satu orang.

Folder ini ada karena konsolidasi kode dilakukan satu orang (Irfan): notebook dari
Bagas, John, dan Sherly diubah jadi `.py` lalu di-push dari satu akun. Akibatnya
riwayat commit GitHub tidak mencerminkan siapa mengerjakan apa, padahal pembagiannya
nyata.

## Isi

| Berkas | Keterangan |
|---|---|
| `kontribusi.md` | Catatan siapa mengerjakan apa, beserta tanggalnya |
| `screenshot/` | Tangkapan layar diskusi dan serah-terima berkas |

## Sebelum menaruh screenshot

Repo ini publik, karena format laporan mensyaratkan tautan bisa dibuka tanpa akses.
Jadi apa pun yang masuk ke sini bisa dibaca siapa saja, dan tidak benar-benar hilang
dari riwayat Git walaupun nanti dihapus.

Tutup dulu bagian ini sebelum menyimpan gambarnya:

- nomor telepon di daftar kontak maupun di header percakapan
- foto profil
- nama kontak yang bukan anggota kelompok
- obrolan di luar topik proyek

Cukup potong bagian isi pesannya saja. Yang perlu terlihat: siapa mengirim apa, dan kapan.

## Catatan

Screenshot adalah bukti pendukung, bukan bukti utama. Tiga hal di bawah lebih kuat dan
tidak butuh sensor apa pun:

1. **Commit atas nama masing-masing.** Kalau mereka bisa pakai Git, satu commit per orang
   sudah cukup untuk memunculkan namanya permanen di daftar kontributor GitHub.
2. **Atribusi saat Irfan yang commit.** Kalau mereka tidak pakai Git, kodenya tetap bisa
   dicatat atas nama penulisnya:
   ```bash
   git commit --author="Bagas Arka Dinanta <email>" -m "Preprocessing dan EDA"
   ```
3. **Notebook asli mereka disimpan apa adanya** di `notebooks/`, berdampingan dengan versi
   `.py` hasil konsolidasi. Output sel di notebook itu hasil eksekusi di mesin mereka, dan
   itu bukti yang jauh lebih sulit dibantah daripada tangkapan layar.

Tabel di `kontribusi.md` juga dipakai ulang sebagai Tabel Pembagian Tugas di Bagian Awal
nomor 4 laporan, jadi mengisinya tidak jadi pekerjaan tambahan.
