# 🏦 Bank Customer Churn Analytics

Aplikasi web interaktif berbasis **Streamlit (Python)** yang mereplikasi secara utuh alur analisis pada panduan **Final Project "Bank Customer Churn" (Datamine Team, DigitalSkola)** menggunakan dataset resmi **`data/Dataset1_Customer_Churn.csv`**. Aplikasi ini dirancang untuk membantu Divisi Marketing perbankan memahami faktor utama pendorong penutupan rekening (*customer churn*), mengevaluasi empat algoritma klasifikasi Machine Learning sebelum dan sesudah penanganan ketidakseimbangan kelas dengan **SMOTE**, serta menyediakan kalkulator prediksi risiko *churn* nasabah secara *real-time*.

---

## 📸 Tampilan Aplikasi (Placeholder Screenshot)

> Silakan ganti bagian di bawah ini dengan tangkapan layar (*screenshot*) aplikasi Anda setelah dijalankan:
>
> - `![Halaman Beranda](docs/screenshot_1_beranda.png)`
> - `![Halaman Exploratory Data Analysis](docs/screenshot_2_eda.png)`
> - `![Halaman Modelling & Evaluasi](docs/screenshot_3_modelling.png)`
> - `![Halaman Prediksi Nasabah](docs/screenshot_4_prediction.png)`

---

## 📊 Spesifikasi Dataset (`data/Dataset1_Customer_Churn.csv`)

Dataset terdiri dari **10.000 baris** dan **7 kolom** (0 *missing value* dan 0 baris duplikat):
- `CustomerId`: ID unik nasabah (dihapus sebelum pemodelan karena tidak relevan terhadap *churn*).
- `Gender`: Jenis kelamin nasabah (`Male` / `Female`).
- `Age`: Usia nasabah dalam satuan tahun.
- `CreditScore`: Nilai skor kredit nasabah berdasarkan riwayat dan perilaku transaksi.
- `EstimatedSalary`: Perkiraan pendapatan atau gaji tahunan nasabah.
- `HasCrCard`: Status kepemilikan kartu kredit (`0` = Tidak punya, `1` = Punya kartu kredit).
- `Exited`: Target status penutupan rekening (`0` = Retained / Bertahan, `1` = Churn / Menutup rekening).

---

## ✨ Fitur Utama & Alur Analisis (8 Halaman)

1. **1. Beranda**
   - Kartu KPI utama yang dihitung langsung dari data: Total Nasabah (`10.000`), Jumlah Churn (`2.037`), Churn Rate (`20,37%`), Rata-rata Usia (`38,9 tahun`), dan Rata-rata Credit Score (`651`).
   - Latar belakang masalah retensi nasabah di Divisi Marketing bank serta tiga *objective* bisnis utama.
2. **2. Data Understanding & Preparation**
   - Validasi otomatis 7 kolom wajib pada `data/Dataset1_Customer_Churn.csv`.
   - Cuplikan tabel `df.head()`, kamus deskripsi tiap kolom, penjelasan alasan `CustomerId` di-*drop*, ringkasan `df.info()`, pemeriksaan *missing value* (`df.isnull().sum()`), serta statistik deskriptif `df.describe()` numerik dan kategorikal.
3. **3. Exploratory Data Analysis (EDA)**
   - *Donut/Pie Chart* proporsi nasabah *Churn* (`20,37%`) vs *Retained* (`79,63%`).
   - *Countplot* variabel kategorikal (`Gender` dan `HasCrCard`) per status `Exited` disertai tabel *churn rate (%)* per kategori dan penjelasan perbedaan antara jumlah absolut vs proporsi.
   - *Boxplot* variabel kontinu (`CreditScore`, `Age`, `EstimatedSalary`) per status `Exited`.
   - *One-Hot Encoding* menggunakan `pd.get_dummies(df, drop_first=True)` sehingga muncul `Gender_Male`.
   - *Heatmap* korelasi **sebelum** dan **sesudah** rekayasa fitur `Credit Score Given Age = CreditScore / Age`, dilengkapi kotak *insight* berbasis angka aktual di bawah setiap visualisasi.
4. **4. Modelling & Evaluasi**
   - Pembagian data `train_test_split(test_size=0.25, random_state=42, stratify=y)` menghasilkan `7.500` baris data latih dan `2.500` baris data uji.
   - Evaluasi 4 algoritma (`GradientBoostingClassifier`, `KNeighborsClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier`) pada skenario **SEBELUM SMOTE** dan **SESUDAH SMOTE** (SMOTE hanya pada data *train* agar bebas *data leakage*).
   - *Selectbox* interaktif untuk membandingkan rasio **SMOTE 1:1 (50:50)** dan **SMOTE ~60:40 (`sampling_strategy=0.6667`)**.
   - *Classification Report*, *Confusion Matrix*, kurva ROC gabungan keempat model pada data *test* (`X_test`), tabel perbandingan performa, penandaan model terbaik berdasarkan **ROC AUC** dan **Recall/F1 kelas churn**, serta penjelasan *trade-off* Recall vs Accuracy.
5. **5. Feature Importance**
   - *Bar chart* horizontal tingkat kepentingan fitur dari model Random Forest beserta tabel perbandingan lintas skenario dan narasi faktor paling berpengaruh.
6. **6. Prediksi Nasabah**
   - Formulir input 5 fitur dasar (`Gender`, `Age`, `CreditScore`, `EstimatedSalary`, `HasCrCard`) dengan perhitungan otomatis fitur `Credit Score Given Age`.
   - Visualisasi *Gauge Chart* probabilitas *churn*, kategori risiko (Rendah / Sedang / Tinggi), dan saran tindakan retensi.
7. **7. Kesimpulan & Rekomendasi**
   - Kesimpulan komprehensif yang menjawab ketiga *objective* bisnis berdasarkan hasil aktual aplikasi, profil risiko kelompok usia, rekomendasi strategi retensi, serta keterbatasan studi (hanya 5 fitur dasar dan ketidakseimbangan kelas).
8. **8. Tentang**
   - Dokumentasi spesifikasi dataset, metodologi analisis DigitalSkola, keterbatasan studi, dan identitas pengembang.

---

## 📂 Struktur Repository

```text
churn-app/
├── app.py                           # Aplikasi utama Streamlit (8 halaman berbahasa Indonesia)
├── train.py                         # Skrip pelatihan offline & penyimpanan artefak model/metrik
├── data/
│   └── Dataset1_Customer_Churn.csv  # Dataset resmi nasabah bank (10.000 baris, 7 kolom)
├── model/
│   ├── rf_pipeline.pkl              # Artefak model Random Forest terlatih (terkompresi)
│   └── metrics.json                 # Artefak metrik evaluasi seluruh model & skenario SMOTE
├── utils/
│   ├── __init__.py
│   ├── data.py                      # Validasi kolom wajib, pemuatan data, persiapan & encoding
│   ├── charts.py                    # Grafik interaktif Plotly bertema perbankan
│   ├── models.py                    # Pelatihan model, evaluasi X_test, SMOTE, dan fungsi prediksi
│   └── style.py                     # Custom CSS, tema kartu KPI, dan komponen header
├── .streamlit/
│   └── config.toml                  # Konfigurasi warna tema perbankan (#0B2545, #F5F7FA, dll.)
├── requirements.txt                 # Daftar dependensi Python dengan versi dipin
├── .gitignore                       # Daftar file/folder yang diabaikan Git
└── README.md                        # Dokumentasi proyek dan panduan deployment
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
   Pastikan seluruh file proyek telah di-*push* ke repository GitHub publik Anda (`afrizalsubagja-unpk/churn-ets-afrizalsubagja-072926002`).
2. **Buka portal Streamlit Community Cloud:**
   Kunjungi [https://share.streamlit.io](https://share.streamlit.io) dan masuk (*sign in*) menggunakan akun GitHub Anda.
3. **Buat aplikasi baru:**
   Klik tombol **"Create app"** (atau **"New app"**) di pojok kanan atas.
4. **Konfigurasi sumber aplikasi:**
   - **Repository:** Pilih `afrizalsubagja-unpk/churn-ets-afrizalsubagja-072926002`.
   - **Branch:** Pilih `main`.
   - **Main file path:** Ketik `app.py`.
5. **Deploy:**
   Klik tombol **"Deploy!"** dan tunggu beberapa saat hingga proses instalasi paket dari `requirements.txt` selesai dan aplikasi Anda tayang secara publik.
