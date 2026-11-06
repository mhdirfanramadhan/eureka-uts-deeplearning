"""Focal loss, fitur unik 1.

Penulis: John Alpha Varensius Purba (231110795).
Dikonversi dari notebooks/02_model_john.ipynb.

Focal loss (Lin dkk., 2017) adalah binary cross-entropy yang dikalikan faktor
(1 - p_t)^gamma, sehingga contoh yang sudah mudah ditebak menyumbang loss jauh
lebih kecil dan pelatihan lebih fokus pada contoh sulit, termasuk kasus stroke
yang langka.

Catatan tentang alpha: di Keras, alpha adalah bobot untuk kelas stroke.
Model yang dilatih dan dievaluasi di laporan ini memakai alpha = 0.25, yaitu
nilai bawaan paper. Percobaan di notebook bagian 7 membandingkan 0.25, 0.5,
dan 0.75 pada data validasi; hasilnya dibahas di laporan A.5e.
"""

import keras

GAMMA = 2.0
ALPHA = 0.25


def focal_loss(gamma=GAMMA, alpha=ALPHA):
    return keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, alpha=alpha, gamma=gamma
    )
