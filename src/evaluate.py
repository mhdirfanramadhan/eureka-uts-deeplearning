import argparse
import json
from pathlib import Path

import numpy as np

BETA = 2.0

NAMA_VERSI = {
    "1_dasar": "Dasar",
    "2_weighted": "Class weight",
    "3_focal": "Focal loss",
    "4_focal_ambang": "Focal + ambang",
}
WARNA = {
    "1_dasar": "#0072B2",
    "2_weighted": "#E69F00",
    "3_focal": "#009E73",
    "4_focal_ambang": "#CC79A7",
}


def _aman(a, b):
    return float(a / b) if b > 0 else 0.0


def muat_npz(jalur):
    d = np.load(jalur)
    return d["X"].astype("float32"), d["y"].astype(int).ravel()


def _rakit_model(keras, config):
    L = keras.layers
    lapis = []
    for ly in config["config"]["layers"]:
        kls, c = ly["class_name"], ly["config"]
        if kls == "InputLayer":
            lapis.append(keras.Input(shape=tuple(c["batch_shape"][1:])))
        elif kls == "Dense":
            reg = c.get("kernel_regularizer")
            if reg and reg["class_name"] != "L2":
                raise ValueError(f"regularizer {reg['class_name']} belum didukung")
            lapis.append(L.Dense(
                c["units"], activation=c.get("activation"), use_bias=c.get("use_bias", True),
                kernel_regularizer=keras.regularizers.L2(reg["config"]["l2"]) if reg else None))
        elif kls == "BatchNormalization":
            lapis.append(L.BatchNormalization(
                axis=c.get("axis", -1), momentum=c.get("momentum", 0.99), epsilon=c.get("epsilon", 1e-3)))
        elif kls == "Activation":
            lapis.append(L.Activation(c["activation"]))
        elif kls == "Dropout":
            lapis.append(L.Dropout(c["rate"]))
        else:
            raise ValueError(f"layer {kls} belum didukung")
    return keras.Sequential(lapis)


def _muat_dari_arsitektur_dan_bobot(keras, jalur):
    import tempfile
    import zipfile

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(jalur) as z:
        config = json.loads(z.read("config.json"))
        z.extract("model.weights.h5", tmp)
    model = _rakit_model(keras, config)
    model.load_weights(str(tmp / "model.weights.h5"))
    return model


def muat_prediktor(jalur):
    import keras

    try:
        model = keras.models.load_model(jalur, compile=False)
    except (TypeError, ValueError):
        try:
            model = _muat_dari_arsitektur_dan_bobot(keras, jalur)
        except Exception as e2:
            raise RuntimeError(
                f"{Path(jalur).name} tidak bisa dimuat dengan Keras {keras.__version__} ({e2}). "
                "Jalankan di Colab (versi Keras sama dengan notebook John) atau perbarui: "
                "pip install -U keras tensorflow"
            ) from e2
    n_fitur = int(model.input_shape[-1])

    def prediksi(X):
        if X.shape[1] != n_fitur:
            raise ValueError(
                f"{Path(jalur).name} dilatih dengan {n_fitur} fitur, tetapi data punya {X.shape[1]}. "
                "Kemungkinan John melatih dengan data sementara. Minta dia menjalankan ulang "
                "notebook-nya dengan PAKAI_DATA_BAGAS = True."
            )
        return model.predict(X.astype("float32"), verbose=0).ravel()

    return prediksi


def hitung_metrik(y, p, ambang):
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    pred = (p >= ambang).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n, n_stroke = len(y), tp + fn
    prec, rec = _aman(tp, tp + fp), _aman(tp, n_stroke)
    return dict(
        ambang=float(ambang),
        precision=prec,
        recall=rec,
        f1=_aman(2 * prec * rec, prec + rec),
        f2=_aman((1 + BETA**2) * prec * rec, BETA**2 * prec + rec),
        pr_auc=float(average_precision_score(y, p)) if n_stroke > 0 else None,
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_total=int(n), n_stroke=int(n_stroke),
        akurasi=_aman(tp + tn, n),
        akurasi_tebak_semua_tidak=_aman(n - n_stroke, n),
        pr_auc_acak=_aman(n_stroke, n),
    )


def _tambal_sklearn():
    import sklearn.compose._column_transformer as ct
    from collections import UserList

    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(UserList):
            pass

        _RemainderColsList.__module__ = ct.__name__
        ct._RemainderColsList = _RemainderColsList


def profil_kesalahan(X, y, pred, jalur_prep):
    try:
        import joblib

        _tambal_sklearn()
        prep = joblib.load(jalur_prep)
        nama = list(prep.get_feature_names_out())
        kolom_num = list(prep.transformers_[0][2])
        idx = [nama.index(f"num__{c}") for c in kolom_num]
        asli = prep.named_transformers_["num"].inverse_transform(X[:, idx])
    except Exception as e:
        kel = {"TP": (y == 1) & (pred == 1), "FN": (y == 1) & (pred == 0),
               "FP": (y == 0) & (pred == 1), "TN": (y == 0) & (pred == 0)}
        z = {k: dict(n=int(m.sum()), **({f"rata2_kolom_{j}": round(float(X[m, j].mean()), 2)
                                         for j in range(X.shape[1])} if m.any() else {}))
             for k, m in kel.items()}
        return {"dilewati": f"preprocessor tidak bisa dipakai: {e}",
                "profil_terstandar": z,
                "catatan": "nilai terstandar (z-score / 0-1), BUKAN umur/glukosa/BMI asli; nomor kolom = urutan fitur di val.npz"}
    flag = {c: nama.index(f"cat__{c}_1") for c in ("hypertension", "heart_disease") if f"cat__{c}_1" in nama}
    kel = {
        "TP": (y == 1) & (pred == 1),
        "FN": (y == 1) & (pred == 0),
        "FP": (y == 0) & (pred == 1),
        "TN": (y == 0) & (pred == 0),
    }
    hasil = {}
    for k, m in kel.items():
        d = {"n": int(m.sum())}
        if m.any():
            for j, c in enumerate(kolom_num):
                d[f"rata2_{c}"] = round(float(asli[m, j].mean()), 2)
            for c, i in flag.items():
                d[f"persen_{c}"] = round(float(X[m, i].mean() * 100), 1)
        hasil[k] = d
    return hasil


def analisis_luput(y, preds):
    stroke = np.flatnonzero(y == 1)
    luput = np.array([[preds[k][i] == 0 for k in preds] for i in stroke])
    jumlah = luput.sum(axis=1)
    n_v = len(preds)
    return dict(
        n_stroke=int(len(stroke)),
        n_versi=n_v,
        luput_di_semua_versi=int((jumlah == n_v).sum()),
        luput_di_sebagian_versi=int(((jumlah > 0) & (jumlah < n_v)).sum()),
        terdeteksi_di_semua_versi=int((jumlah == 0).sum()),
        indeks_baris_test_luput_di_semua_versi=[int(i) for i in stroke[jumlah == n_v]],
    )


def ringkas_overfitting(log):
    hasil = {}
    for nama, m in log["model"].items():
        h = m["history"]
        loss, val = np.array(h["loss"]), np.array(h["val_loss"])
        eb = int(m["epoch_terbaik"]) - 1
        i_min = int(np.argmin(val))
        naik = _aman(val[-1] - val[i_min], val[i_min]) * 100
        latih_turun = bool(loss[-1] < loss[i_min])
        d = dict(
            epoch_jalan=int(len(loss)),
            epoch_terbaik=eb + 1,
            loss_latih_di_epoch_terbaik=float(loss[eb]),
            loss_val_di_epoch_terbaik=float(val[eb]),
            selisih_val_minus_latih_di_epoch_terbaik=float(val[eb] - loss[eb]),
            epoch_val_loss_terendah=i_min + 1,
            val_loss_terendah=float(val[i_min]),
            val_loss_akhir=float(val[-1]),
            kenaikan_val_loss_akhir_dari_terendah_persen=round(float(naik), 1),
            loss_latih_masih_turun_sampai_akhir=latih_turun,
            indikasi_overfitting=bool(naik > 10 and latih_turun),
        )
        if m.get("class_weight"):
            d["catatan"] = ("loss latih dikali bobot kelas, loss validasi tidak: selisih keduanya "
                            "tidak sebanding dengan model lain")
        hasil[nama] = d
    return hasil


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def gambar_loss(log, jalur):
    plt = _plt()
    model = log["model"]
    fig, axes = plt.subplots(1, len(model), figsize=(5 * len(model), 4))
    for ax, (nama, m) in zip(np.atleast_1d(axes), model.items()):
        h = m["history"]
        x = np.arange(1, len(h["loss"]) + 1)
        ax.plot(x, h["loss"], label="latih", color="#0072B2")
        ax.plot(x, h["val_loss"], label="validasi", color="#D55E00")
        ax.axvline(m["epoch_terbaik"], color="gray", ls="--", lw=1, label="epoch terbaik")
        judul = f"model_{nama}" + (" (loss latih berbobot)" if m.get("class_weight") else "")
        ax.set_title(judul, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(jalur, dpi=150)
    plt.close(fig)


def gambar_pr(y, p_model, hasil, jalur):
    from sklearn.metrics import precision_recall_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, k in (("dasar", "1_dasar"), ("weighted", "2_weighted"), ("focal", "3_focal")):
        pr, rc, _ = precision_recall_curve(y, p_model[m])
        ax.plot(rc, pr, color=WARNA[k], lw=1.4, alpha=0.8, label=f"kurva model_{m}")
    for k, h in hasil.items():
        ax.scatter(h["recall"], h["precision"], s=70, color=WARNA[k], edgecolor="black", zorder=3,
                   label=f"{NAMA_VERSI[k]} (ambang {h['ambang']:.3g})")
    ax.axhline(hasil["1_dasar"]["pr_auc_acak"], color="gray", ls=":", label="model acak")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Kurva precision-recall, data test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150)
    plt.close(fig)


def gambar_cm(hasil, jalur):
    plt = _plt()
    fig, axes = plt.subplots(1, len(hasil), figsize=(4 * len(hasil), 3.8))
    for ax, (k, h) in zip(np.atleast_1d(axes), hasil.items()):
        cm = np.array([[h["tn"], h["fp"]], [h["fn"], h["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["tidak stroke", "stroke"])
        ax.set_yticks([0, 1], ["tidak stroke", "stroke"])
        ax.set_xlabel("Tebakan")
        ax.set_ylabel("Sebenarnya")
        ax.set_title(f"{NAMA_VERSI[k]}\nrecall {h['tp']}/{h['n_stroke']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150)
    plt.close(fig)


def gambar_perbandingan(hasil, jalur):
    plt = _plt()
    metrik = ["recall", "precision", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    lebar = 0.2
    for i, (k, h) in enumerate(hasil.items()):
        nilai = [h[m] for m in metrik]
        xs = np.arange(len(metrik)) + (i - 1.5) * lebar
        ax.bar(xs, nilai, lebar, color=WARNA[k], label=NAMA_VERSI[k])
        ax.text(xs[0], nilai[0] + 0.01, f"{h['tp']}/{h['n_stroke']}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(metrik)), ["recall", "precision", "F1", "PR-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan empat versi, data test (angka di atas recall = pasien terdeteksi)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150)
    plt.close(fig)


def jalankan(data="data/processed", art="artifacts", fig="figures", prep=None, force=False,
             prediktor=muat_prediktor):
    data, art, fig = Path(data), Path(art), Path(fig)
    prep = Path(prep) if prep else art / "preprocessor.joblib"
    keluar = art / "metrics.json"
    if keluar.exists() and not force:
        raise SystemExit(
            f"{keluar} sudah ada, jadi data test sudah pernah dibuka. Menjalankan ulang setelah melihat "
            "hasil berarti memakai data test untuk memperbaiki model. Pakai --force hanya kalau "
            "proses sebelumnya gagal di tengah."
        )
    jalur_ambang = art / "threshold.json"
    if not jalur_ambang.exists():
        raise SystemExit("threshold.json belum ada. Jalankan src/threshold.py dulu (hanya memakai data validasi).")
    cfg = json.loads(jalur_ambang.read_text())
    fig.mkdir(parents=True, exist_ok=True)

    X, y = muat_npz(data / "test.npz")

    p = {m: prediktor(art / f"model_{m}.keras")(X) for m in sorted({v["model"] for v in cfg["versi"].values()})}

    sumber = cfg["versi"][cfg["versi_final"]]["model"]
    p_final = prediktor(art / "model_final.keras")(X)
    salinan_sama = bool(np.allclose(p_final, p[sumber], atol=1e-6))
    if not salinan_sama:
        print("PERINGATAN: model_final.keras tidak sama dengan", f"model_{sumber}.keras")

    hasil = {}
    for k, v in cfg["versi"].items():
        h = hitung_metrik(y, p[v["model"]], v["ambang"])
        h.update(model=f"model_{v['model']}.keras", nama=NAMA_VERSI[k])
        hasil[k] = h
    dasar = hasil["1_dasar"]
    for k, h in hasil.items():
        d = h["tp"] - dasar["tp"]
        h["selisih_vs_dasar"] = None if k == "1_dasar" else dict(
            pasien_stroke_terdeteksi=d,
            tambahan_salah_alarm=h["fp"] - dasar["fp"],
            selisih_pr_auc=h["pr_auc"] - dasar["pr_auc"],
            bermakna=abs(d) >= 1,
        )
    f3, f4 = hasil["3_focal"], hasil["4_focal_ambang"]
    f4["selisih_vs_focal_ambang_0_5"] = dict(
        pasien_stroke_terdeteksi=f4["tp"] - f3["tp"],
        tambahan_salah_alarm=f4["fp"] - f3["fp"],
        bermakna=abs(f4["tp"] - f3["tp"]) >= 1,
    )

    pred = {k: (p[v["model"]] >= v["ambang"]).astype(int) for k, v in cfg["versi"].items()}
    kesalahan = dict(
        versi_dianalisis_profil=cfg["versi_final"],
        luput=analisis_luput(y, pred),
        profil=profil_kesalahan(X, y, pred[cfg["versi_final"]], prep),
    )

    overfit = None
    jalur_log = art / "train_log.json"
    if jalur_log.exists():
        log = json.loads(jalur_log.read_text())
        overfit = ringkas_overfitting(log)
        gambar_loss(log, fig / "loss_latih_vs_validasi.png")

    gambar_pr(y, p, hasil, fig / "kurva_pr_test.png")
    gambar_cm(hasil, fig / "confusion_matrix_test.png")
    gambar_perbandingan(hasil, fig / "perbandingan_versi_test.png")

    metrik = dict(
        catatan=("Semua angka laporan diambil dari berkas ini. Data test dibuka satu kali. Ambang dan versi "
                 "final ditentukan dari data validasi sebelum data test dibuka. PR-AUC dihitung dengan "
                 "average_precision_score (scikit-learn), jadi bisa sedikit beda dari val_auc_pr milik Keras."),
        data_test=dict(n=int(len(y)), n_stroke=int(y.sum())),
        versi_final=cfg["versi_final"],
        model_final_sama_dengan_sumber=salinan_sama,
        versi=hasil,
        kesalahan=kesalahan,
        overfitting=overfit,
        validasi_dari_threshold_json=cfg.get("validasi"),
    )
    keluar.write_text(json.dumps(metrik, indent=2, ensure_ascii=False))

    print(f"\nData test: {len(y)} pasien, {int(y.sum())} stroke\n")
    print(f"{'Versi':<16}{'ambang':>8}{'recall':>9}{'stroke':>9}{'prec':>8}{'F1':>7}{'PR-AUC':>8}{'FP':>6}")
    for k, h in hasil.items():
        print(f"{NAMA_VERSI[k]:<16}{h['ambang']:>8.3f}{h['recall']:>9.3f}"
              f"{h['tp']:>5}/{h['n_stroke']:<3}{h['precision']:>8.3f}{h['f1']:>7.3f}{h['pr_auc']:>8.3f}{h['fp']:>6}")
    print(f"\nAkurasi 'tebak semua tidak stroke' = {dasar['akurasi_tebak_semua_tidak']:.3f} "
          f"(PR-AUC acak = {dasar['pr_auc_acak']:.3f})")
    print("Tersimpan:", keluar)
    return metrik


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluasi di data test (dibuka satu kali).")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--fig", default="figures")
    ap.add_argument("--prep", default=None, help="preprocessor.joblib (default: <art>/preprocessor.joblib)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jalankan(a.data, a.art, a.fig, a.prep, a.force)
