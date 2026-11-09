import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import BETA, NAMA_VERSI, WARNA, hitung_metrik, muat_npz, muat_prediktor


def kurva_ambang(y, p, beta=BETA):
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("data tidak punya pasien stroke, ambang tidak bisa dicari")
    urut = np.argsort(-p, kind="stable")
    ps, ys = p[urut], y[urut]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    idx = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    tp, fp = tp[idx], fp[idx]
    berikut = np.r_[ps[idx[:-1] + 1], ps[-1]]
    ambang = (ps[idx] + berikut) / 2
    prec = tp / (tp + fp)
    rec = tp / n_pos
    b2 = beta**2
    fb = np.divide((1 + b2) * prec * rec, b2 * prec + rec, out=np.zeros_like(prec), where=(b2 * prec + rec) > 0)
    return dict(ambang=ambang, precision=prec, recall=rec, f_beta=fb, tp=tp, fp=fp, n_stroke=n_pos, n=len(y))


def cari_ambang(y, p, beta=BETA, presisi_min=0.0):
    k = kurva_ambang(y, p, beta)
    layak = (k["tp"] > 0) & (k["precision"] >= presisi_min)
    if not layak.any():
        raise ValueError("tidak ada ambang yang memenuhi presisi_min")
    skor = np.where(layak, k["f_beta"], -1.0)
    kandidat = np.flatnonzero(np.isclose(skor, skor.max()))
    i = int(kandidat[len(kandidat) // 2])
    b2 = beta**2
    pr_semua = k["n_stroke"] / k["n"]
    f_semua = (1 + b2) * pr_semua / (b2 * pr_semua + 1)
    hasil = dict(
        ambang=float(k["ambang"][i]),
        beta=float(beta),
        f_beta=float(skor[i]),
        precision=float(k["precision"][i]),
        recall=float(k["recall"][i]),
        tp=int(k["tp"][i]),
        fp=int(k["fp"][i]),
        jumlah_ambang_seri=int(len(kandidat)),
        f_beta_jika_semua_ditebak_stroke=float(f_semua),
    )
    if hasil["f_beta"] <= f_semua:
        hasil["peringatan"] = "F-beta terbaik tidak melebihi tebak-semua-stroke: model hampir tidak berguna"
    return hasil, k


def gambar_ambang(k, terpilih, jalur):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    o = np.argsort(k["ambang"])
    ax.plot(k["ambang"][o], k["recall"][o], label="recall", color="#0072B2")
    ax.plot(k["ambang"][o], k["precision"][o], label="precision", color="#D55E00")
    ax.plot(k["ambang"][o], k["f_beta"][o], label=f"F{terpilih['beta']:g}", color="#009E73", lw=2)
    ax.axvline(0.5, color="gray", ls=":", label="bawaan 0,5")
    ax.axvline(terpilih["ambang"], color="black", ls="--", label=f"terpilih {terpilih['ambang']:.3f}")
    ax.set_xlabel("Ambang")
    ax.set_ylabel("Nilai (data validasi)")
    ax.set_title("Mencari ambang untuk model_focal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(jalur, dpi=150)
    plt.close(fig)


def jalankan(data="data/processed", art="artifacts", fig="figures", prediktor=muat_prediktor):
    data, art, fig = Path(data), Path(art), Path(fig)
    fig.mkdir(parents=True, exist_ok=True)

    Xv, yv = muat_npz(data / "val.npz")
    p_val = {m: prediktor(art / f"model_{m}.keras")(Xv) for m in ("dasar", "weighted", "focal")}

    cari, kurva = cari_ambang(yv, p_val["focal"])
    versi = {
        "1_dasar": ("dasar", 0.5),
        "2_weighted": ("weighted", 0.5),
        "3_focal": ("focal", 0.5),
        "4_focal_ambang": ("focal", cari["ambang"]),
    }
    val = {k: hitung_metrik(yv, p_val[m], t) for k, (m, t) in versi.items()}

    final = max(val, key=lambda k: (val[k]["f2"], val[k]["pr_auc"]))
    shutil.copyfile(art / f"model_{versi[final][0]}.keras", art / "model_final.keras")

    cfg = dict(
        ambang=versi[final][1],
        versi_final=final,
        model_final="model_final.keras",
        sumber_model=f"model_{versi[final][0]}.keras",
        dicari_dari="data validasi",
        aturan_pemilihan="F2 validasi tertinggi, seri dipecah PR-AUC validasi",
        pencarian=cari,
        versi={k: dict(model=m, ambang=float(t)) for k, (m, t) in versi.items()},
        validasi=val,
        data_validasi=dict(n=int(len(yv)), n_stroke=int(yv.sum())),
    )
    (art / "threshold.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    gambar_ambang(kurva, cari, fig / "ambang_validasi.png")

    print(f"Data validasi: {len(yv)} pasien, {int(yv.sum())} stroke")
    print(f"Ambang terpilih untuk model_focal: {cari['ambang']:.4f} "
          f"(F{BETA:g} {cari['f_beta']:.3f}, recall {cari['tp']}/{int(yv.sum())}, precision {cari['precision']:.3f})")
    if "peringatan" in cari:
        print("PERINGATAN:", cari["peringatan"])
    print(f"\n{'Versi':<16}{'ambang':>8}{'recall':>9}{'prec':>8}{'F2':>7}{'PR-AUC':>8}")
    for k, h in val.items():
        print(f"{NAMA_VERSI[k]:<16}{h['ambang']:>8.3f}{h['recall']:>9.3f}{h['precision']:>8.3f}"
              f"{h['f2']:>7.3f}{h['pr_auc']:>8.3f}")
    print(f"\nVersi final: {NAMA_VERSI[final]} -> artifacts/model_final.keras, ambang {cfg['ambang']:.4f}")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cari ambang dari data validasi.")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    jalankan(a.data, a.art, a.fig)
