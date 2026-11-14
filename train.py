"""
Skrip pelatihan offline untuk menghasilkan model/rf_pipeline.pkl dan model/metrics.json.
Jalankan dengan perintah:
    python train.py
"""

import sys
from utils.data import load_raw_dataset
from utils.models import train_and_evaluate_all, PIPELINE_PATH, METRICS_PATH


def main() -> None:
    """Memuat dataset, melatih seluruh model sebelum & sesudah SMOTE, serta menyimpan artefak."""
    print("=" * 68)
    print("  PELATIHAN MODEL BANK CUSTOMER CHURN ANALYTICS (DigitalSkola)")
    print("=" * 68)

    try:
        df_raw, ignored_cols = load_raw_dataset("data/Churn_Modelling.csv")
    except Exception as exc:
        print(f"[ERROR] Gagal memuat dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Dataset berhasil dimuat: {len(df_raw):,} baris, {len(df_raw.columns)} kolom wajib.")
    if ignored_cols:
        print(f"[INFO] Kolom tambahan yang diabaikan sesuai panduan: {', '.join(ignored_cols)}")

    print("[INFO] Melatih 4 model (Gradient Boosting, KNN, Random Forest, Decision Tree)...")
    _, metrics = train_and_evaluate_all(df_raw, save_artifacts=True)

    split_info = metrics["split_info"]
    print(f"[INFO] Ukuran Data Train : {split_info['train_rows']:,} baris")
    print(f"[INFO] Ukuran Data Test  : {split_info['test_rows']:,} baris")

    print("\n--- Ringkasan ROC AUC pada Data Test (X_test) ---")
    for scen_key, scen_title in [
        ("before_smote", "Sebelum SMOTE"),
        ("after_smote_1_1", "Sesudah SMOTE (1:1)"),
        ("after_smote_60_40", "Sesudah SMOTE (60:40)"),
    ]:
        print(f"\n[{scen_title}]")
        for m_name, m_val in metrics[scen_key].items():
            print(
                f"  - {m_name:<22} | Accuracy: {m_val['accuracy']:.4f} | "
                f"Recall (Churn): {m_val['recall_churn']:.4f} | "
                f"F1 (Churn): {m_val['f1_churn']:.4f} | ROC AUC: {m_val['roc_auc']:.4f}"
            )

    best = metrics["best_overall"]
    print("\n" + "=" * 68)
    print(
        f"[SUKSES] Model Terbaik (ROC AUC): {best['model_name']} ({best['scenario']}) "
        f"-> AUC = {best['roc_auc']:.4f}"
    )
    print(f"[SUKSES] Artefak model disimpan di : {PIPELINE_PATH}")
    print(f"[SUKSES] Artefak metrik disimpan di: {METRICS_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
