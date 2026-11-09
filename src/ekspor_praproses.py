import json
from pathlib import Path

import joblib
import numpy as np

AKAR = Path(__file__).resolve().parents[1]
ART = AKAR / "artifacts"
KELUARAN = ART / "praproses.json"


def main():
    pre = joblib.load(ART / "preprocessor.joblib")

    num_cols, mean, scale = None, None, None
    cat_cols, kategori, drop_idx = None, {}, None

    for nama, trans, kolom in pre.transformers_:
        jenis = type(trans).__name__
        if jenis == "StandardScaler":
            num_cols = list(kolom)
            mean = [float(v) for v in trans.mean_]
            scale = [float(v) for v in trans.scale_]
        elif jenis == "OneHotEncoder":
            cat_cols = list(kolom)
            drop_idx = [int(v) for v in trans.drop_idx_]
            for k, c in zip(cat_cols, trans.categories_):
                kategori[k] = [v.item() if hasattr(v, "item") else v for v in c]
        else:
            raise SystemExit(f"transformer belum didukung: {jenis}")

    if drop_idx is not None and set(drop_idx) != {0}:
        raise SystemExit(f"drop_idx selain 0 belum didukung: {drop_idx}")

    data = {
        "num_cols": num_cols,
        "mean": mean,
        "scale": scale,
        "cat_cols": cat_cols,
        "kategori": kategori,
        "buang_kategori_pertama": True,
        "nama_fitur": [str(v) for v in pre.get_feature_names_out()],
    }

    KELUARAN.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(num_cols) + sum(len(v) - 1 for v in kategori.values())
    print(f"fitur numerik    : {len(num_cols)}")
    print(f"fitur kategorikal: {len(cat_cols)}")
    print(f"total fitur      : {n}  (preprocessor: {len(data['nama_fitur'])})")
    print(f"ukuran berkas    : {KELUARAN.stat().st_size / 1024:.1f} KB")
    print(f"tersimpan        : {KELUARAN}")

    assert n == len(data["nama_fitur"])
    print("\nOK")


if __name__ == "__main__":
    main()
