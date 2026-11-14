# 🏦 Bank Customer Churn Analytics

Aplikasi web interaktif berbasis **Streamlit (Python)** yang mereplikasi secara utuh alur analisis pada panduan **Final Project "Bank Customer Churn" (Datamine Team, DigitalSkola)**. Aplikasi ini dirancang untuk membantu Divisi Marketing perbankan memahami faktor utama pendorong penutupan rekening (*customer churn*), mengevaluasi empat algoritma klasifikasi Machine Learning sebelum dan sesudah penanganan ketidakseimbangan kelas dengan **SMOTE**, serta menyediakan kalkulator prediksi risiko *churn* nasabah secara *real-time*.

---

## 📸 Tampilan Aplikasi (Placeholder Screenshot)

> Silakan ganti bagian di bawah ini dengan tangkapan layar (*screenshot*) aplikasi Anda setelah dijalankan:
>
> - `![Halaman Business Understanding](docs/screenshot_1_business.png)`
> - `![Halaman Exploratory Data Analysis](docs/screenshot_2_eda.png)`
> - `![Halaman Modelling & Evaluation](docs/screenshot_3_modelling.png)`
> - `![Halaman Prediksi Nasabah](docs/screenshot_4_prediction.png)`

---

## ✨ Fitur Utama & Alur Analisis (8 Halaman)

1. **1. Business Understanding**
   - Latar belakang masalah retensi nasabah di Divisi Marketing bank.
   - Kartu KPI utama yang dihitung langsung dari data: Total Nasabah (`10.000`), Jumlah Churn (`2.037`), Churn Rate (`20,37%`), Rata-rata Usia (`38,9 tahun`), dan Rata-rata Credit Score (`651`).
   - Penjelasan tiga tujuan utama (*Business & Analytical Objectives*).
2. **2. Data Understanding**
   - Validasi otomatis 7 kolom wajib (`CustomerId`, `Gender`, `Age`, `CreditScore`, `EstimatedSalary`, `HasCrCard`, `Exited`) dan pengabaian kolom di luar panduan.
   - Tabel interaktif `df.head()`, dimensi dataset, serta kamus deskripsi setiap variabel.
3. **3. Data Preparation**
   - Penghapusan kolom `CustomerId` yang tidak relevan terhadap pemodelan.
   - Tabel ringkasan `df.info()` (tipe data & jumlah non-null), pemeriksaan *missing value* (`df.isnull().sum()`), serta statistik deskriptif `df.describe()` untuk atribut numerik dan kategorikal.
4. **4. Exploratory Data Analysis (EDA) & Feature Engineering**
   - *Donut/Pie Chart* proporsi nasabah *Churn* vs *Retained*.
   - *Countplot* variabel kategorikal (`HasCrCard` dan `Gender`) terhadap status `Exited` beserta tabel *churn rate (%)* per kategori.
   - *Boxplot* variabel kontinu (`CreditScore`, `Age`, `EstimatedSalary`) per status `Exited`.
   - *One-Hot Encoding* menggunakan `pd.get_dummies(df, drop_first=True)` menghasilkan `Gender_Male`.
   - *Correlation Matrix (Heatmap)* **sebelum** dan **sesudah** rekayasa fitur `Credit Score Given Age = CreditScore / Age`, dilengkapi kotak *insight* berbasis angka aktual di bawah setiap visualisasi.
5. **5. Modelling & Evaluation**
   - Pembagian data `train_test_split(test_size=0.25, random_state=42, stratify=y)` menghasilkan `7.500` baris data latih dan `2.500` baris data uji.
   - Evaluasi 4 algoritma (`GradientBoostingClassifier`, `KNeighborsClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier`) pada **Skenario A (Sebelum SMOTE)** dan **Skenario B (Sesudah SMOTE pada data train)** tanpa *data leakage*.
   - *Classification Report*, *Confusion Matrix*, kurva ROC gabungan keempat model, tabel perbandingan performa, analisis *trade-off Recall vs Accuracy*, serta *Feature Importance* dari Random Forest.
6. **6. Kesimpulan, Rekomendasi & Saran**
   - Jawaban komprehensif atas ketiga *objective* berdasarkan angka aktual aplikasi.
   - Komparasi interaktif saran panduan: rasio **SMOTE 1:1 (50:50)** vs **SMOTE 60:40 (`sampling_strategy=0.6667`)**.
7. **7. Prediksi Nasabah**
   - Formulir input profil nasabah dengan perhitungan otomatis fitur `Credit Score Given Age`.
   - Visualisasi *Gauge Chart* probabilitas *churn*, klasifikasi kategori risiko (Rendah / Sedang / Tinggi), dan daftar rekomendasi tindakan retensi.
8. **8. Tentang**
   - Ringkasan metodologi, sumber panduan DigitalSkola, keterbatasan dataset, dan placeholder identitas penyusun (`[Nama Anda]` / `[NIM]`).

---

## 📂 Struktur Repository

```text
churn-app/
├── app.py                     # Aplikasi utama Streamlit (8 halaman berbahasa Indonesia)
├── train.py                   # Skrip pelatihan offline & penyimpanan artefak model/metrik
├── data/
│   └── Churn_Modelling.csv    # Dataset nasabah bank (10.000 baris)
├── model/
│   ├── rf_pipeline.pkl        # Artefak model Random Forest terlatih
│   └── metrics.json           # Artefak metrik evaluasi seluruh model & skenario SMOTE
├── utils/
│   ├── __init__.py
│   ├── data.py                # Validasi kolom wajib, pemuatan data, persiapan & encoding
│   ├── charts.py              # Grafik interaktif Plotly bertema perbankan
│   ├── models.py              # Pelatihan model, evaluasi X_test, SMOTE, dan fungsi prediksi
│   └── style.py               # Custom CSS, tema kartu KPI, dan komponen header
├── .streamlit/
│   └── config.toml            # Konfigurasi warna tema perbankan (#0B2545, #F5F7FA, dll.)
├── requirements.txt           # Daftar dependensi Python dengan versi dipin
├── .gitignore                 # Daftar file/folder yang diabaikan Git
└── README.md                  # Dokumentasi proyek dan panduan deployment
```

---

## 💻 Cara Menjalankan Secara Lokal

Pastikan Anda telah menginstal **Python 3.10+** di komputer Anda, lalu jalankan perintah berikut di terminal:

```bash
# 1. Instal seluruh dependensi yang diperlukan
pip install -r requirements.txt

# 2. Jalankan skrip pelatihan untuk menghasilkan model/rf_pipeline.pkl dan model/metrics.json
python train.py

# 3. Jalankan aplikasi web Streamlit
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di peramban pada alamat `http://localhost:8501`.

---

## 🚀 Langkah Deployment ke Streamlit Community Cloud

1. **Push kode ke repository GitHub publik:**
   Pastikan seluruh file proyek (termasuk `app.py`, `train.py`, `requirements.txt`, `.streamlit/config.toml`, folder `data/`, `model/`, dan `utils/`) telah di-*push* ke repository GitHub publik Anda.
2. **Buka portal Streamlit Community Cloud:**
   Kunjungi [https://share.streamlit.io](https://share.streamlit.io) dan masuk (*sign in*) menggunakan akun GitHub Anda.
3. **Buat aplikasi baru:**
   Klik tombol **"Create app"** (atau **"New app"**) di pojok kanan atas.
4. **Konfigurasi sumber aplikasi:**
   - **Repository:** Pilih repository GitHub publik Anda (`afrizalsubagja-unpk/churn-ets-afrizalsubagja-072926002`).
   - **Branch:** Pilih `main`.
   - **Main file path:** Ketik `app.py`.
5. **Deploy:**
   Klik tombol **"Deploy!"** dan tunggu beberapa saat hingga proses instalasi paket dari `requirements.txt` selesai dan aplikasi Anda tayang secara publik.
# churn-ets-afrizalsubagja-072926002
