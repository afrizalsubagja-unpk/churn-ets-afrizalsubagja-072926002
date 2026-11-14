"""Modul pelatihan model Machine Learning, evaluasi metrik, SMOTE, dan prediksi nasabah."""

import json
import os
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.data import encode_data, engineer_features, prepare_data

MODEL_DIR = "model"
PIPELINE_PATH = os.path.join(MODEL_DIR, "rf_pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def _get_base_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Mengembalikan 4 algoritma klasifikasi sesuai panduan Final Project."""
    return {
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }


def _evaluate_single_model(
    clf: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, Any], Any]:
    """
    Melatih satu model pada data train dan mengevaluasinya pada data test (X_test).
    Menghitung classification report, confusion matrix, dan ROC AUC dari predict_proba(X_test).
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec_churn = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_churn = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_churn = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # Downsample titik kurva ROC jika terlalu panjang agar ukuran JSON ringkas dan cepat dimuat
    if len(fpr) > 250:
        indices = np.linspace(0, len(fpr) - 1, 250, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]

    result = {
        "accuracy": acc,
        "precision_churn": prec_churn,
        "recall_churn": rec_churn,
        "f1_churn": f1_churn,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "fpr": [round(float(x), 5) for x in fpr],
        "tpr": [round(float(x), 5) for x in tpr],
    }

    if hasattr(clf, "feature_importances_"):
        result["feature_importances"] = {
            col: float(imp) for col, imp in zip(X_train.columns, clf.feature_importances_)
        }

    return result, clf


def train_and_evaluate_all(
    df_raw: pd.DataFrame, save_artifacts: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Menjalankan seluruh alur persiapan fitur, pembagian train/test (test_size=0.25, stratify=y, random_state=42),
    dan melatih 4 model untuk:
      1. Skenario A: Sebelum SMOTE (Data Train Asli)
      2. Skenario B: Sesudah SMOTE 1:1 (sampling_strategy=1.0 pada data train)
      3. Skenario C (Saran): Sesudah SMOTE ~60:40 (sampling_strategy=2/3 pada data train)

    Menyimpan artefak model (`model/rf_pipeline.pkl`) dan metrik (`model/metrics.json`).
    """
    df_prep = prepare_data(df_raw)
    df_encoded = encode_data(df_prep)
    df_fe = engineer_features(df_encoded)

    feature_cols: List[str] = [col for col in df_fe.columns if col != "Exited"]
    X = df_fe[feature_cols]
    y = df_fe["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Skenario A: Sebelum SMOTE
    models_before: Dict[str, Any] = {}
    trained_rf_before = None
    for name, clf in _get_base_classifiers(random_state=42).items():
        res, fitted = _evaluate_single_model(clf, X_train, y_train, X_test, y_test)
        models_before[name] = res
        if name == "Random Forest":
            trained_rf_before = fitted

    # Skenario B: Sesudah SMOTE 1:1 (50:50) hanya pada data train
    smote_1_1 = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_sm1, y_train_sm1 = smote_1_1.fit_resample(X_train, y_train)

    models_after_1_1: Dict[str, Any] = {}
    trained_rf_smote_1_1 = None
    for name, clf in _get_base_classifiers(random_state=42).items():
        res, fitted = _evaluate_single_model(clf, X_train_sm1, y_train_sm1, X_test, y_test)
        models_after_1_1[name] = res
        if name == "Random Forest":
            trained_rf_smote_1_1 = fitted

    # Skenario C (Saran Panduan): SMOTE rasio 60:40 (kelas minoritas = 40/60 = 0.6667 dari mayoritas)
    smote_60_40 = SMOTE(sampling_strategy=round(40 / 60, 4), random_state=42)
    X_train_sm60, y_train_sm60 = smote_60_40.fit_resample(X_train, y_train)

    models_after_60_40: Dict[str, Any] = {}
    trained_rf_smote_60_40 = None
    for name, clf in _get_base_classifiers(random_state=42).items():
        res, fitted = _evaluate_single_model(clf, X_train_sm60, y_train_sm60, X_test, y_test)
        models_after_60_40[name] = res
        if name == "Random Forest":
            trained_rf_smote_60_40 = fitted

    # Pilih model Random Forest terbaik berdasarkan ROC AUC untuk prediksi nasabah
    rf_candidates = {
        "Sebelum SMOTE": (models_before["Random Forest"]["roc_auc"], trained_rf_before),
        "Sesudah SMOTE (1:1)": (models_after_1_1["Random Forest"]["roc_auc"], trained_rf_smote_1_1),
        "Sesudah SMOTE (60:40)": (models_after_60_40["Random Forest"]["roc_auc"], trained_rf_smote_60_40),
    }
    best_rf_scenario = max(rf_candidates.keys(), key=lambda k: rf_candidates[k][0])

    # Cari model terbaik secara keseluruhan (dari 4 algoritma x 2 skenario utama) berdasarkan ROC AUC
    all_main_entries = []
    for m_name, m_val in models_before.items():
        all_main_entries.append(("Sebelum SMOTE", m_name, m_val["roc_auc"], m_val["recall_churn"], m_val["accuracy"]))
    for m_name, m_val in models_after_1_1.items():
        all_main_entries.append(("Sesudah SMOTE (1:1)", m_name, m_val["roc_auc"], m_val["recall_churn"], m_val["accuracy"]))

    best_overall = max(all_main_entries, key=lambda item: item[2])

    metrics_payload: Dict[str, Any] = {
        "split_info": {
            "total_rows": int(len(df_fe)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "num_features": int(len(feature_cols)),
            "feature_columns": feature_cols,
            "train_class_dist_before": {
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum()),
            },
            "test_class_dist": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
            "train_class_dist_smote_1_1": {
                "0": int((y_train_sm1 == 0).sum()),
                "1": int((y_train_sm1 == 1).sum()),
            },
            "train_class_dist_smote_60_40": {
                "0": int((y_train_sm60 == 0).sum()),
                "1": int((y_train_sm60 == 1).sum()),
            },
        },
        "before_smote": models_before,
        "after_smote_1_1": models_after_1_1,
        "after_smote_60_40": models_after_60_40,
        "best_overall": {
            "scenario": best_overall[0],
            "model_name": best_overall[1],
            "roc_auc": best_overall[2],
            "recall_churn": best_overall[3],
            "accuracy": best_overall[4],
        },
        "best_rf_scenario": best_rf_scenario,
    }

    pipeline_bundle: Dict[str, Any] = {
        "feature_columns": feature_cols,
        "best_rf_scenario": best_rf_scenario,
        "rf_before_smote": trained_rf_before,
        "rf_after_smote_1_1": trained_rf_smote_1_1,
        "rf_after_smote_60_40": trained_rf_smote_60_40,
    }

    if save_artifacts:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline_bundle, PIPELINE_PATH, compress=3)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

    return pipeline_bundle, metrics_payload


@st.cache_resource(show_spinner="Memuat model Machine Learning & metrik evaluasi...")
def load_or_train_artifacts(df_raw: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Memuat model (`model/rf_pipeline.pkl`) dan metrik (`model/metrics.json`) jika sudah tersedia.
    Apabila salah satu file tidak ditemukan atau rusak, otomatis melatih ulang dari data aktual.
    """
    if os.path.exists(PIPELINE_PATH) and os.path.exists(METRICS_PATH):
        try:
            pipeline_bundle = joblib.load(PIPELINE_PATH)
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_payload = json.load(f)
            return pipeline_bundle, metrics_payload
        except Exception:
            pass

    return train_and_evaluate_all(df_raw, save_artifacts=True)


def build_comparison_dataframe(metrics_payload: Dict[str, Any], include_60_40: bool = False) -> pd.DataFrame:
    """
    Membangun DataFrame ringkasan perbandingan performa keempat model
    (Accuracy, Precision, Recall, F1 kelas churn, ROC AUC) pada data test.
    """
    rows = []
    scenarios = [
        ("Sebelum SMOTE", metrics_payload["before_smote"]),
        ("Sesudah SMOTE (1:1)", metrics_payload["after_smote_1_1"]),
    ]
    if include_60_40 and "after_smote_60_40" in metrics_payload:
        scenarios.append(("Sesudah SMOTE (60:40)", metrics_payload["after_smote_60_40"]))

    for scen_label, scen_models in scenarios:
        for model_name, m_data in scen_models.items():
            rows.append(
                {
                    "Skenario": scen_label,
                    "Model": model_name,
                    "Accuracy": round(m_data["accuracy"], 4),
                    "Precision (Churn)": round(m_data["precision_churn"], 4),
                    "Recall (Churn)": round(m_data["recall_churn"], 4),
                    "F1-Score (Churn)": round(m_data["f1_churn"], 4),
                    "ROC AUC": round(m_data["roc_auc"], 4),
                }
            )
    return pd.DataFrame(rows)


def format_classification_report_df(report_dict: Dict[str, Any]) -> pd.DataFrame:
    """Mengubah dictionary classification_report menjadi DataFrame yang mudah dibaca."""
    label_mapping = {
        "0": "Kelas 0 (Retained)",
        "1": "Kelas 1 (Churn)",
        "macro avg": "Macro Average",
        "weighted avg": "Weighted Average",
    }
    records = []
    for key, label in label_mapping.items():
        if key in report_dict and isinstance(report_dict[key], dict):
            item = report_dict[key]
            records.append(
                {
                    "Kelas / Rata-rata": label,
                    "Precision": round(float(item.get("precision", 0.0)), 4),
                    "Recall": round(float(item.get("recall", 0.0)), 4),
                    "F1-Score": round(float(item.get("f1-score", 0.0)), 4),
                    "Support (Jumlah Data)": int(item.get("support", 0)),
                }
            )
    return pd.DataFrame(records)


def predict_customer_churn(
    pipeline_bundle: Dict[str, Any],
    gender: str,
    age: int,
    credit_score: int,
    estimated_salary: float,
    has_cr_card: int,
    rf_mode: str = "Sesudah SMOTE (60:40)",
) -> Dict[str, Any]:
    """
    Menghitung fitur rekayasa 'Credit Score Given Age' secara otomatis dan memprediksi
    probabilitas churn nasabah menggunakan model Random Forest terpilih.
    """
    if age <= 0:
        raise ValueError("Usia nasabah harus lebih besar dari 0 tahun.")

    cs_given_age = float(credit_score) / float(age)
    gender_male = 1 if gender == "Male" else 0

    feature_cols: List[str] = pipeline_bundle["feature_columns"]
    row_dict = {
        "CreditScore": float(credit_score),
        "Age": float(age),
        "HasCrCard": int(has_cr_card),
        "EstimatedSalary": float(estimated_salary),
        "Gender_Male": int(gender_male),
        "Credit Score Given Age": cs_given_age,
    }
    X_input = pd.DataFrame([[row_dict[col] for col in feature_cols]], columns=feature_cols)

    model_map = {
        "Sebelum SMOTE": pipeline_bundle["rf_before_smote"],
        "Sesudah SMOTE (1:1)": pipeline_bundle["rf_after_smote_1_1"],
        "Sesudah SMOTE (60:40)": pipeline_bundle["rf_after_smote_60_40"],
    }
    clf = model_map.get(rf_mode, pipeline_bundle["rf_after_smote_60_40"])

    prob_churn = float(clf.predict_proba(X_input)[0, 1])
    pred_label = int(clf.predict(X_input)[0])

    if prob_churn < 0.35:
        risk_category = "Rendah"
        risk_color = "#1B998B"
        recommendations = [
            "Pertahankan kualitas layanan perbankan digital dan komunikasi berkala.",
            "Tawarkan program loyalitas reguler atau *cross-selling* produk tabungan berjangka.",
            "Lakukan pemantauan rutin tahunan tanpa perlu intervensi diskon agresif.",
        ]
    elif prob_churn < 0.65:
        risk_category = "Sedang"
        risk_color = "#F4B942"
        recommendations = [
            "Berikan penawaran *reward points* atau bebas biaya administrasi bulanan untuk menjaga keaktifan.",
            "Kirimkan survei kepuasan singkat untuk mengidentifikasi potensi kendala layanan sejak dini.",
            "Tawarkan edukasi atau penyesuaian produk finansial yang sesuai dengan kelompok usia nasabah.",
        ]
    else:
        risk_category = "Tinggi"
        risk_color = "#E4572E"
        recommendations = [
            "Segera jadwalkan kontak proaktif oleh *Relationship Manager* untuk menanyakan kebutuhan nasabah.",
            "Berikan insentif retensi khusus (misalnya *cashback* transaksi, bunga tabungan spesial, atau restrukturisasi fasilitas kredit).",
            "Prioritaskan penanganan keluhan nasabah melalui jalur layanan prioritas (*fast-track retention*). ",
        ]

    return {
        "prob_churn": prob_churn,
        "prob_retained": 1.0 - prob_churn,
        "pred_label": pred_label,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "credit_score_given_age": cs_given_age,
        "input_dataframe": X_input,
        "recommendations": recommendations,
    }
