import streamlit as st

from predictor import ambang, prediksi, siap

st.set_page_config(page_title="Prediksi Risiko Stroke", layout="centered")

st.title("Prediksi Risiko Stroke")
st.caption("Tim Eureka · Pembelajaran Mendalam")

if not siap():
    st.warning(
        "Model asli belum terpasang. Angka di bawah masih contoh, bukan hasil prediksi."
    )

with st.form("pasien"):
    kiri, kanan = st.columns(2)

    with kiri:
        gender = st.selectbox("Jenis kelamin", ["Male", "Female"])
        age = st.number_input("Umur (tahun)", 0, 100, 50, step=1)
        hypertension = st.selectbox("Riwayat hipertensi", [0, 1])
        heart_disease = st.selectbox("Riwayat penyakit jantung", [0, 1])
        ever_married = st.selectbox("Pernah menikah", ["Yes", "No"])

    with kanan:
        work_type = st.selectbox(
            "Jenis pekerjaan",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence = st.selectbox("Tempat tinggal", ["Urban", "Rural"])
        glukosa = st.number_input("Rata-rata glukosa (mg/dL)", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
        smoking = st.selectbox(
            "Status merokok",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    kirim = st.form_submit_button("Hitung risiko")

if kirim:
    pasien = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glukosa,
        "bmi": bmi,
        "smoking_status": smoking,
    }

    p = prediksi(pasien)
    t = ambang()

    st.metric("Peluang stroke", f"{p:.1%}")
    st.caption(f"Ambang yang dipakai: {t:.2f}")

    if p >= t:
        st.error("Masuk kelompok berisiko pada ambang ini. Sebaiknya diperiksa lebih lanjut.")
    else:
        st.success("Tidak masuk kelompok berisiko pada ambang ini.")

    st.caption(
        "Ini bukan diagnosis medis. Hasilnya berasal dari model statistik yang "
        "dilatih pada data publik, bukan pengganti pemeriksaan dokter."
    )
