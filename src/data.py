import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
AKAR = Path(__file__).resolve().parents[1]
DIR_PROC = AKAR / "data" / "processed"
DIR_ART = AKAR / "artifacts"
DIR_FIG = AKAR / "figures"

NUM_COLS = ["age", "avg_glucose_level", "bmi"]
CAT_COLS = ["gender", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "smoking_status"]


def muat(csv):
    df = pd.read_csv(csv)
    print(f"Data dimuat. Total baris: {df.shape[0]}, total kolom: {df.shape[1]}")
    return df


def inspeksi(df):
    print("\n=== 1. DISTRIBUSI TARGET STROKE ===")
    c = df["stroke"].value_counts()
    print(c)
    print(f"Total: {len(df)} baris, kasus stroke: {c[1]} ({c[1] / len(df) * 100:.2f}%)")

    print("\n=== 2. DISTRIBUSI GENDER ===")
    print(df["gender"].value_counts())

    print("\n=== 3. ANALISIS MISSING VALUE BMI ===")
    null_bmi = df[df["bmi"].isnull()]
    print(f"Total baris BMI kosong: {len(null_bmi)}")
    print(f"Pasien stroke dengan BMI kosong: {null_bmi['stroke'].sum()} orang")

    print("\n=== 4. ANALISIS SMOKING STATUS ===")
    print(df["smoking_status"].value_counts())
    pct = df["smoking_status"].value_counts()["Unknown"] / len(df) * 100
    print(f"Persentase Unknown: {pct:.2f}%")


def gambar_eda(df):
    DIR_FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.countplot(x="stroke", data=df, ax=axes[0], hue="stroke",
                  palette="Set2", legend=False)
    axes[0].set_title("Distribusi Imbalance Target")
    sns.histplot(data=df, x="age", hue="stroke", kde=True, ax=axes[1], palette="Set1")
    axes[1].set_title("Distribusi Umur vs Stroke")
    sns.histplot(data=df, x="bmi", hue="stroke", kde=True, ax=axes[2], palette="Set1")
    axes[2].set_title("Distribusi BMI vs Stroke")
    plt.tight_layout()
    plt.savefig(DIR_FIG / "eda_distribusi.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 6))
    num = df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriks Korelasi Antarfitur Numerik")
    plt.tight_layout()
    plt.savefig(DIR_FIG / "eda_korelasi.png", dpi=120)
    plt.close()
    print(f"\nGambar EDA disimpan ke {DIR_FIG}")


def bersihkan(df):
    d = df.drop(columns=["id"]).copy()
    d = d[d["gender"] != "Other"]
    d["bmi"] = d["bmi"].fillna(d["bmi"].median())
    X = d.drop(columns=["stroke"])
    y = d["stroke"]
    print(f"\nPembersihan selesai. X: {X.shape}, y: {y.shape}")
    return X, y


def bagi(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    print("\n=== STRATIFIED SPLIT 70/15/15 ===")
    for nama, Xs, ys in (("Train", X_tr, y_tr), ("Val", X_va, y_va), ("Test", X_te, y_te)):
        print(f"{nama:6s}: {Xs.shape[0]:5d} baris | pasien stroke: {ys.sum()}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def preprocess(X_tr, X_va, X_te):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
    ])
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    X_te_p = pre.transform(X_te)
    print(f"\nPreprocessing selesai. Dimensi data latih akhir: {X_tr_p.shape}")
    return pre, X_tr_p, X_va_p, X_te_p


def simpan(pre, X_tr_p, X_va_p, X_te_p, y_tr, y_va, y_te):
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    DIR_ART.mkdir(parents=True, exist_ok=True)
    np.savez(DIR_PROC / "train.npz", X=X_tr_p, y=y_tr.values)
    np.savez(DIR_PROC / "val.npz", X=X_va_p, y=y_va.values)
    np.savez(DIR_PROC / "test.npz", X=X_te_p, y=y_te.values)
    joblib.dump(pre, DIR_ART / "preprocessor.joblib")
    print("\n=== BERKAS DIEKSPOR ===")
    for f in ("data/processed/train.npz", "data/processed/val.npz",
              "data/processed/test.npz", "artifacts/preprocessor.joblib"):
        print(" ", f)


def main(csv):
    df = muat(csv)
    inspeksi(df)
    gambar_eda(df)
    X, y = bersihkan(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = bagi(X, y)
    pre, a, b, c = preprocess(X_tr, X_va, X_te)
    simpan(pre, a, b, c, y_tr, y_va, y_te)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else AKAR / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
    main(csv)
