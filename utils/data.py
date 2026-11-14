"""Modul pemuatan, validasi, dan pemrosesan data Bank Customer Churn."""

import os
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

DEFAULT_DATA_PATH = "data/Dataset1_Customer_Churn.csv"

# Daftar 7 kolom wajib pada Dataset1_Customer_Churn.csv
REQUIRED_COLUMNS: List[str] = [
    "CustomerId",
    "Gender",
    "Age",
    "CreditScore",
    "EstimatedSalary",
    "HasCrCard",
    "Exited",
]

# Deskripsi masing-masing kolom untuk halaman Data Understanding
COLUMN_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "CustomerId": ("Numerik (ID)", "ID unik yang dimiliki oleh masing-masing nasabah bank."),
    "Gender": ("Kategorikal", "Jenis kelamin nasabah (Female = Perempuan, Male = Laki-laki)."),
    "Age": ("Numerik (Kontinu)", "Usia nasabah dalam satuan tahun."),
    "CreditScore": (
        "Numerik (Kontinu)",
        "Nilai skor kredit nasabah berdasarkan riwayat dan perilaku transaksi perbankan.",
    ),
    "EstimatedSalary": (
        "Numerik (Kontinu)",
        "Perkiraan pendapatan atau gaji tahunan nasabah.",
    ),
    "HasCrCard": (
        "Kategorikal (Biner)",
        "Status kepemilikan kartu kredit (1 = Memiliki kartu kredit, 0 = Tidak memiliki).",
    ),
    "Exited": (
        "Target (Biner)",
        "Status penutupan rekening nasabah (1 = Churn / Menutup rekening, 0 = Retained / Bertahan).",
    ),
}


def load_raw_dataset(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV dan memvalidasi keberadaan 7 kolom wajib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File dataset tidak ditemukan pada jalur: '{filepath}'.")

    df_full = pd.read_csv(filepath)

    # Cek kelengkapan kolom wajib
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_full.columns]
    if missing_cols:
        raise ValueError(
            f"Kolom wajib berikut tidak ditemukan di dalam dataset '{filepath}': {', '.join(missing_cols)}"
        )

    return df_full[REQUIRED_COLUMNS].copy()


@st.cache_data(show_spinner=False)
def get_validated_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Fungsi ter-cache untuk memuat data di Streamlit.
    Jika file hilang atau kolom wajib kurang, tampilkan st.error dan hentikan aplikasi dengan st.stop().
    """
    try:
        return load_raw_dataset(filepath)
    except FileNotFoundError as err:
        st.error(
            f"🚨 **File Dataset Tidak Ditemukan!**\n\n{err}\n\n"
            f"Pastikan file `{DEFAULT_DATA_PATH}` tersedia di dalam repositori proyek."
        )
        st.stop()
    except ValueError as err:
        st.error(
            f"🚨 **Validasi Kolom Wajib Gagal!**\n\n{err}\n\n"
            f"Dataset wajib memuat ketujuh kolom berikut: `{', '.join(REQUIRED_COLUMNS)}`."
        )
        st.stop()
    except Exception as err:
        st.error(f"🚨 **Terjadi Kesalahan Saat Memuat Data:**\n\n{err}")
        st.stop()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahap Data Preparation: menghapus kolom CustomerId karena tidak relevan terhadap prediksi churn.
    """
    if "CustomerId" in df.columns:
        return df.drop(columns=["CustomerId"]).copy()
    return df.copy()


def encode_data(df_prep: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding pada variabel kategorikal menggunakan pd.get_dummies(drop_first=True)
    sehingga terbentuk kolom Gender_Male (1 = Male, 0 = Female).
    """
    df_encoded = pd.get_dummies(df_prep, drop_first=True)
    for col in df_encoded.select_dtypes(include=["bool"]).columns:
        df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


def engineer_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan Feature Engineering dengan menambahkan fitur baru:
    'Credit Score Given Age' = CreditScore / Age.
    """
    df_fe = df_encoded.copy()
    df_fe["Credit Score Given Age"] = df_fe["CreditScore"] / df_fe["Age"]
    return df_fe


def get_info_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan ringkasan struktur DataFrame setara df.info() dalam bentuk tabel yang rapi.
    """
    info_records = []
    total_rows = len(df)
    for idx, col in enumerate(df.columns):
        non_null = int(df[col].notnull().sum())
        null_count = int(df[col].isnull().sum())
        info_records.append(
            {
                "No": idx + 1,
                "Nama Kolom": col,
                "Jumlah Non-Null": f"{non_null:,} non-null",
                "Missing Value (Null)": null_count,
                "Persentase Lengkap (%)": round((non_null / total_rows) * 100, 2) if total_rows > 0 else 0.0,
                "Tipe Data (dtype)": str(df[col].dtype),
                "Jumlah Nilai Unik": int(df[col].nunique()),
            }
        )
    return pd.DataFrame(info_records)


def compute_kpi_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Menghitung metrik utama (KPI) secara dinamis dari dataset.
    """
    total_customers = int(len(df))
    churn_count = int((df["Exited"] == 1).sum())
    retained_count = int((df["Exited"] == 0).sum())
    churn_rate = float((churn_count / total_customers) * 100.0) if total_customers > 0 else 0.0
    avg_age = float(df["Age"].mean())
    avg_credit_score = float(df["CreditScore"].mean())
    avg_salary = float(df["EstimatedSalary"].mean())
    cr_card_pct = float((df["HasCrCard"] == 1).mean() * 100.0)
    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df.isnull().sum().sum())

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": churn_rate,
        "retained_rate": 100.0 - churn_rate,
        "avg_age": avg_age,
        "avg_credit_score": avg_credit_score,
        "avg_salary": avg_salary,
        "cr_card_pct": cr_card_pct,
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
    }
