import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Judul Aplikasi (Sesuai tema SDG 3)
st.set_page_config(page_title="Deteksi Dini Alzheimer - SDG 3", layout="centered")
st.title("🧠 Aplikasi Deteksi Dini Penyakit Alzheimer")
st.markdown("### **Mendukung SDG Target 3.4:** Mendorong Kesehatan Mental & Mengurangi Dampak PTM")
st.write("Masukkan indikator klinis dan gaya hidup pasien di bawah ini untuk melihat hasil prediksi.")

# 2. Memuat Model, Scaler, dan Kolom Fitur yang telah dilatih
@st.cache_resource
def load_models():
    model = joblib.load('model_alzheimer.pkl')
    scaler = joblib.load('scaler_alzheimer.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, scaler, features

try:
    model, scaler, feature_columns = load_models()
except:
    st.error("Gagal memuat model. Pastikan Anda telah menjalankan 'train_model.py' terlebih dahulu!")
    st.stop()

# 3. Membuat Input Form Antarmuka Pengguna (UI)
st.sidebar.header("📋 Data Demografis & Gaya Hidup")
age = st.sidebar.slider("Usia (Tahun)", 60, 90, 75)
gender = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Pria" if x == 0 else "Wanita")
bmi = st.sidebar.slider("Indeks Massa Tubuh (BMI)", 15.0, 40.0, 24.5)
smoking = st.sidebar.selectbox("Status Merokok", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
physical_activity = st.sidebar.slider("Aktivitas Fisik (Jam/Minggu)", 0.0, 10.0, 3.5)
sleep_quality = st.sidebar.slider("Skor Kualitas Tidur", 4.0, 10.0, 7.0)

st.header("🩺 Hasil Tes Klinis & Kognitif")
col1, col2 = st.columns(2)

with col1:
    mmse = st.slider("Skor Tes Kognitif (MMSE)", 0, 30, 20)
    functional_assessment = st.slider("Skor Fungsional Mandiri", 0.0, 10.0, 5.0)
    depression = st.selectbox("Riwayat Depresi", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

with col2:
    memory_complaints = st.selectbox("Keluhan Gangguan Ingatan", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    behavioral_problems = st.selectbox("Masalah Perubahan Perilaku", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    forgetfulness = st.selectbox("Sifat Pelupa yang Tidak Wajar", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

# 4. Melakukan Prediksi Saat Tombol Ditekan
if st.button("Mulai Deteksi Risiko", type="primary"):
    
    # Membuat dictionary default untuk mencocokkan seluruh struktur dataset asli
    input_dict = {col: 0 for col in feature_columns}
    
    # Mengisi dictionary dengan input dari user
    input_dict['Age'] = age
    input_dict['Gender'] = gender
    input_dict['BMI'] = bmi
    input_dict['Smoking'] = smoking
    input_dict['PhysicalActivity'] = physical_activity
    input_dict['SleepQuality'] = sleep_quality
    input_dict['MMSE'] = mmse
    input_dict['FunctionalAssessment'] = functional_assessment
    input_dict['Depression'] = depression
    input_dict['MemoryComplaints'] = memory_complaints
    input_dict['BehavioralProblems'] = behavioral_problems
    input_dict['Forgetfulness'] = forgetfulness
    
    # Konversi ke DataFrame sesuai urutan kolom fitur aslinya
    input_df = pd.DataFrame([input_dict])[feature_columns]
    
    # Standarisasi data input menggunakan scaler dari training
    input_scaled = scaler.transform(input_df)
    
    # Prediksi menggunakan model Random Forest
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 5. Menampilkan Hasil Prediksi ke Layar
    st.markdown("---")
    st.subheader("📊 Hasil Pemeriksaan Komputasi AI")
    
    if prediction == 1:
        st.error(f"⚠️ **Hasil Prediksi: Berisiko Positif Terdiagnosis Alzheimer**")
        st.write(f"Probabilitas Risiko: **{probabilities[1] * 100:.2f}%**")
        st.info("💡 **Rekomendasi SDG 3:** Segera lakukan konsultasi lanjutan dengan dokter spesialis saraf atau psikiater untuk intervensi kesehatan mental dini.")
    else:
        st.success(f"✅ **Hasil Prediksi: Negatif (Kondisi Kognitif Cenderung Stabil)**")
        st.write(f"Probabilitas Risiko: **{probabilities[1] * 100:.2f}%**")
        st.info("💡 **Rekomendasi SDG 3:** Pertahankan pola aktivitas fisik, kualitas tidur, dan stimulasi otak berkala guna menjaga Well-Being di hari tua.")
