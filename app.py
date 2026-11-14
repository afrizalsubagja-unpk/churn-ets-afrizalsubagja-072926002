"""
Aplikasi Web Streamlit: Bank Customer Churn Analytics
Mereplikasi alur analisis panduan Final Project "Bank Customer Churn" (Datamine Team, DigitalSkola)
menggunakan dataset data/Dataset1_Customer_Churn.csv.
"""

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from utils.charts import (
    plot_categorical_countplot,
    plot_churn_pie,
    plot_churn_probability_gauge,
    plot_combined_roc_curve,
    plot_confusion_matrix,
    plot_continuous_boxplot,
    plot_correlation_heatmap,
    plot_feature_importance,
)
from utils.data import (
    COLUMN_DESCRIPTIONS,
    DEFAULT_DATA_PATH,
    REQUIRED_COLUMNS,
    compute_kpi_summary,
    encode_data,
    engineer_features,
    get_info_dataframe,
    get_validated_data,
    prepare_data,
)
from utils.models import (
    build_comparison_dataframe,
    format_classification_report_df,
    load_or_train_artifacts,
    predict_customer_churn,
)
from utils.style import (
    COLORS,
    apply_custom_style,
    render_insight_box,
    render_kpi_card,
    render_page_header,
    render_section_title,
    render_top_kpi_row,
)

# ---------------------------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_style()

# ---------------------------------------------------------------------------
# PEMUATAN DATA & VALIDASI KOLOM WAJIB (data/Dataset1_Customer_Churn.csv)
# ---------------------------------------------------------------------------
df_raw = get_validated_data(DEFAULT_DATA_PATH)
kpi_summary = compute_kpi_summary(df_raw)

# Tahapan pemrosesan sesuai urutan panduan
df_prep = prepare_data(df_raw)
df_encoded = encode_data(df_prep)
df_fe = engineer_features(df_encoded)

# Muat model dan hasil evaluasi dari model/ atau latih otomatis jika belum ada
pipeline_bundle, metrics_payload = load_or_train_artifacts(df_raw)

# ---------------------------------------------------------------------------
# NAVIGASI SIDEBAR (8 HALAMAN SESUAI REVISI)
# ---------------------------------------------------------------------------
MENU_OPTIONS = [
    "1. Beranda",
    "2. Data Understanding & Preparation",
    "3. Exploratory Data Analysis",
    "4. Modelling & Evaluasi",
    "5. Feature Importance",
    "6. Prediksi Nasabah",
    "7. Kesimpulan & Rekomendasi",
    "8. Tentang",
]

MENU_ICONS = [
    "house-door-fill",
    "table",
    "bar-chart-line-fill",
    "cpu-fill",
    "Award-fill" if False else "diagram-3-fill",
    "person-check-fill",
    "check2-circle",
    "info-circle-fill",
]

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 0.6rem 0.4rem 1rem 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 0.8rem;">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COLORS['gold']} !important; font-weight: 700;">
                DASHBOARD ANALITIK PERBANKAN
            </div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF !important; margin-top: 0.2rem;">
                🏦 Bank Churn Analytics
            </div>
            <div style="font-size: 0.78rem; color: #D1D5DB !important; margin-top: 0.15rem;">
                Metodologi Datamine Team • DigitalSkola
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = option_menu(
        menu_title=None,
        options=MENU_OPTIONS,
        icons=MENU_ICONS,
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": COLORS["gold"], "font-size": "15px"},
            "nav-link": {
                "font-size": "13.2px",
                "text-align": "left",
                "margin": "3px 0px",
                "padding": "9px 12px",
                "border-radius": "8px",
                "color": "#F3F4F6",
                "--hover-color": "rgba(255, 255, 255, 0.1)",
            },
            "nav-link-selected": {
                "background-color": COLORS["teal"],
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    st.markdown(
        f"""
        <div style="margin-top: 1.2rem; padding: 0.85rem; background: rgba(255,255,255,0.06); border-radius: 10px; font-size: 0.78rem;">
            <div style="font-weight: 700; color: {COLORS['gold']} !important; margin-bottom: 0.35rem;">
                📌 Ringkasan Dataset Aktif
            </div>
            <div>• File: <code>Dataset1_Customer_Churn.csv</code></div>
            <div>• Total Data: <b>{kpi_summary['total_customers']:,}</b> baris</div>
            <div>• Jumlah Kolom: <b>{len(REQUIRED_COLUMNS)}</b> kolom</div>
            <div>• Churn Rate: <b>{kpi_summary['churn_rate']:.2f}%</b></div>
            <div>• Model Terbaik: <b>{metrics_payload['best_overall']['model_name']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HALAMAN 1: BERANDA (BUSINESS UNDERSTANDING & KPI CARDS)
# ===========================================================================
if selected_page == "1. Beranda":
    render_page_header(
        title="1. Beranda — Business Understanding",
        subtitle=(
            "Ringkasan eksekutif indikator kinerja utama (KPI) populasi nasabah, latar belakang masalah penutupan "
            "rekening (churn), serta tiga tujuan utama analisis data perbankan."
        ),
        badge="Tahap 1 • Beranda & Pemahaman Bisnis",
    )

    # Baris KPI Utama (Total Nasabah, Jumlah Churn, Churn Rate, Rata-rata Usia, Rata-rata CreditScore)
    render_section_title("Indikator Kinerja Utama (KPI) Populasi Nasabah")
    render_top_kpi_row(kpi_summary)

    # Latar Belakang Masalah
    render_section_title("Latar Belakang Bisnis")
    col_bg1, col_bg2 = st.columns([1.55, 1])
    with col_bg1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>🏦 Tantangan Divisi Marketing Perbankan</h4>
                <p>
                    Divisi Marketing dari sebuah bank memiliki sasaran strategis untuk terus <b>menambah nasabah baru</b>
                    sekaligus <b>mempertahankan nasabah yang sudah ada (retensi)</b> agar tidak menutup rekening mereka
                    (<i>customer churn</i>). Apabila tingkat <i>churn</i> nasabah tinggi, profitabilitas bank akan menurun
                    secara signifikan dan menghambat laju pertumbuhan bisnis jangka panjang.
                </p>
                <p style="margin-top: 0.7rem;">
                    Berdasarkan data aktual pada <code>{DEFAULT_DATA_PATH}</code> sebanyak <b>{kpi_summary['total_customers']:,} nasabah</b>,
                    tercatat sejumlah <b style="color:{COLORS['danger']};">{kpi_summary['churn_count']:,} nasabah ({kpi_summary['churn_rate']:.2f}%)</b>
                    telah menutup rekeningnya (<i>Exited = 1</i>), sementara
                    <b style="color:{COLORS['teal']};">{kpi_summary['retained_count']:,} nasabah ({kpi_summary['retained_rate']:.2f}%)</b>
                    tetap bertahan (<i>Exited = 0</i>). Biaya akuisisi nasabah baru di industri perbankan jauh lebih tinggi
                    dibandingkan biaya mempertahankan nasabah eksisting, sehingga pendekatan berbasis <i>Machine Learning</i>
                    sangat krusial untuk mendeteksi potensi <i>churn</i> lebih dini.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_bg2:
        avg_age_churn = float(df_raw[df_raw["Exited"] == 1]["Age"].mean())
        avg_age_ret = float(df_raw[df_raw["Exited"] == 0]["Age"].mean())
        female_rate = float(df_raw[df_raw["Gender"] == "Female"]["Exited"].mean() * 100.0)
        male_rate = float(df_raw[df_raw["Gender"] == "Male"]["Exited"].mean() * 100.0)
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {COLORS['gold']};">
                <h4>📊 Potret Awal Risiko Churn dari Data Aktual</h4>
                <p>
                    • <b>Rasio Kelas Target:</b> <b>{kpi_summary['retained_rate']:.2f}% Bertahan</b> vs <b>{kpi_summary['churn_rate']:.2f}% Churn</b>.<br>
                    • <b>Usia Rata-rata Nasabah Churn:</b> <b>{avg_age_churn:.1f} tahun</b> (lebih tua dibandingkan nasabah bertahan yaitu <b>{avg_age_ret:.1f} tahun</b>).<br>
                    • <b>Churn Rate per Gender:</b> Perempuan <b>{female_rate:.2f}%</b> vs Laki-laki <b>{male_rate:.2f}%</b>.<br>
                    • <b>Kepemilikan Kartu Kredit:</b> <b>{kpi_summary['cr_card_pct']:.2f}%</b> nasabah memiliki kartu kredit.<br>
                    • <b>Rata-rata Gaji Tahunan:</b> <b>${kpi_summary['avg_salary']:,.2f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tiga Objective Utama
    render_section_title("Tiga Tujuan Utama (Business & Analytical Objectives)")
    obj_cols = st.columns(3)
    with obj_cols[0]:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['navy']}; height: 100%;">
                <h4>🎯 Objective (a): Faktor Pendorong Churn</h4>
                <p>
                    Mengidentifikasi <b>faktor atau atribut apa yang paling memengaruhi tingginya tingkat <i>churn</i></b>
                    nasabah melalui analisis eksplorasi proporsi kategori, matriks korelasi, rekayasa fitur
                    (<code>Credit Score Given Age</code>), serta evaluasi <i>Feature Importance</i>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with obj_cols[1]:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['teal']}; height: 100%;">
                <h4>🤖 Objective (b): Model Prediksi Paling Akurat</h4>
                <p>
                    Membandingkan performa empat algoritma klasifikasi (<i>Gradient Boosting, KNN, Random Forest, Decision Tree</i>)
                    baik <b>sebelum maupun sesudah SMOTE</b> pada data uji untuk menentukan model mana yang paling akurat dan sensitif
                    memprediksi nasabah <i>churn</i>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with obj_cols[2]:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['gold']}; height: 100%;">
                <h4>📈 Objective (c): Strategi Pertumbuhan Bisnis</h4>
                <p>
                    Merumuskan bagaimana bank dapat <b>memanfaatkan hasil analisis dan model prediktif ini</b> untuk menyusun
                    program retensi terfokus pada segmen berisiko tinggi, menekan kehilangan nasabah, dan meningkatkan
                    pertumbuhan bisnis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# HALAMAN 2: DATA UNDERSTANDING & PREPARATION
# ===========================================================================
elif selected_page == "2. Data Understanding & Preparation":
    render_page_header(
        title="2. Data Understanding & Data Preparation",
        subtitle=(
            "Eksplorasi struktur dataset Dataset1_Customer_Churn.csv (head, info, deskripsi 7 kolom), pemeriksaan "
            "missing value & duplikat, statistik deskriptif numerik/kategorikal, serta penghapusan CustomerId."
        ),
        badge="Tahap 2 • Pemahaman & Persiapan Data",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Jumlah Baris Data", f"{len(df_raw):,}", "Observasi nasabah bank", "navy")
    with c2:
        render_kpi_card("Jumlah Kolom", f"{len(df_raw.columns)} Kolom", "6 kolom setelah drop CustomerId", "teal")
    with c3:
        render_kpi_card("Missing Value", f"{kpi_summary['missing_values']}", "Tidak ada nilai kosong (0%)", "blue")
    with c4:
        render_kpi_card("Data Duplikat", f"{kpi_summary['duplicate_rows']}", "Seluruh baris unik (100%)", "gold")

    # 1. Tabel head() dan Deskripsi Tiap Kolom
    render_section_title("1. Cuplikan Tabel Data Awal — df.head() & Deskripsi 7 Kolom")
    num_rows_preview = st.slider(
        "Pilih jumlah baris awal yang ingin ditampilkan:",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
    st.dataframe(df_raw.head(num_rows_preview), use_container_width=True, hide_index=False)

    desc_rows = []
    for col_name in REQUIRED_COLUMNS:
        col_type, col_desc = COLUMN_DESCRIPTIONS.get(col_name, ("-", "-"))
        sample_vals = df_raw[col_name].dropna().unique()[:4]
        sample_str = ", ".join(str(v) for v in sample_vals)
        desc_rows.append(
            {
                "Nama Kolom": col_name,
                "Tipe Data (dtype)": str(df_raw[col_name].dtype),
                "Jenis Variabel": col_type,
                "Contoh Nilai": sample_str,
                "Deskripsi Kolom": col_desc,
            }
        )
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

    # 2. Penjelasan alasan CustomerId di-drop
    render_section_title("2. Data Preparation: Penghapusan Kolom CustomerId — df.drop('CustomerId')")
    col_p1, col_p2 = st.columns([1.2, 1])
    with col_p1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>🧹 Mengapa Kolom <code>CustomerId</code> Dihapus Sebelum Modeling?</h4>
                <p>
                    Kolom <code>CustomerId</code> berisi nomor identitas unik untuk setiap nasabah (terdapat
                    <b>{df_raw['CustomerId'].nunique():,} nilai unik</b> dari {len(df_raw):,} baris). Atribut ini bersifat administratif
                    semata dan tidak memiliki makna perilaku maupun hubungan kausal terhadap keputusan nasabah menutup rekening (<i>churn</i>).
                </p>
                <p style="margin-top:0.6rem;">
                    Apabila <code>CustomerId</code> dimasukkan ke dalam model <i>Machine Learning</i>, algoritma pohon keputusan dapat
                    menghafal nomor ID tertentu (<i>overfitting</i>) dan menurunkan kemampuan generalisasi pada nasabah baru.<br><br>
                    • <b>Kolom Awal ({len(df_raw.columns)} kolom):</b> <code>{', '.join(df_raw.columns)}</code><br>
                    • <b>Kolom Setelah Preparation ({len(df_prep.columns)} kolom):</b> <code>{', '.join(df_prep.columns)}</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2:
        st.markdown("**Cuplikan Data Setelah `CustomerId` Dihapus (`df_prep.head()`):**")
        st.dataframe(df_prep.head(6), use_container_width=True)

    # 3. Struktur info() & Cek Missing Value
    render_section_title("3. Pemeriksaan Tipe Data (df.info()) & Missing Value (df.isnull().sum())")
    info_df = get_info_dataframe(df_prep)
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    render_insight_box(
        f"Seluruh <b>{len(df_prep.columns)} kolom aktif</b> pada <b>{len(df_prep):,} baris data</b> memiliki "
        f"<b>0 missing value</b> (<code>isnull().sum() = 0</code>) dan <b>0 baris duplikat</b>, terdiri dari 5 kolom numerik "
        f"(<code>Age, CreditScore, EstimatedSalary, HasCrCard, Exited</code>) serta 1 kolom kategorikal (<code>Gender</code>)."
    )

    # 4. Statistik Deskriptif Numerik & Kategorikal
    render_section_title("4. Statistik Deskriptif Numerik & Kategorikal — df.describe()")
    tab_num, tab_cat = st.tabs(["🔢 Statistik Numerik (df.describe())", "🔤 Statistik Kategorikal (include='object')"])
    with tab_num:
        desc_num = df_prep.describe().T.reset_index().rename(columns={"index": "Atribut Numerik"})
        for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if c in desc_num.columns:
                desc_num[c] = desc_num[c].round(2)
        st.dataframe(desc_num, use_container_width=True, hide_index=True)
        render_insight_box(
            f"Usia nasabah (<b>Age</b>) berkisar antara <b>{int(df_prep['Age'].min())} – {int(df_prep['Age'].max())} tahun</b> "
            f"(rata-rata <b>{df_prep['Age'].mean():.2f} tahun</b>), skor kredit (<b>CreditScore</b>) berada di rentang "
            f"<b>{int(df_prep['CreditScore'].min())} – {int(df_prep['CreditScore'].max())}</b> (rata-rata <b>{df_prep['CreditScore'].mean():.2f}</b>), "
            f"dan perkiraan gaji (<b>EstimatedSalary</b>) berkisar antara <b>${df_prep['EstimatedSalary'].min():,.2f} – ${df_prep['EstimatedSalary'].max():,.2f}</b> "
            f"(rata-rata <b>${df_prep['EstimatedSalary'].mean():,.2f}</b>)."
        )
    with tab_cat:
        desc_cat = df_prep.describe(include=["object"]).T.reset_index().rename(columns={"index": "Atribut Kategorikal"})
        st.dataframe(desc_cat, use_container_width=True, hide_index=True)
        top_gender = str(desc_cat.loc[0, "top"])
        freq_gender = int(desc_cat.loc[0, "freq"])
        female_cnt = int((df_prep["Gender"] == "Female").sum())
        render_insight_box(
            f"Variabel <b>Gender</b> terdiri dari 2 nilai unik: <b>{top_gender}</b> sebanyak <b>{freq_gender:,} nasabah ({freq_gender / len(df_prep) * 100:.2f}%)</b> "
            f"dan <b>Female</b> sebanyak <b>{female_cnt:,} nasabah ({female_cnt / len(df_prep) * 100:.2f}%)</b>."
        )


# ===========================================================================
# HALAMAN 3: EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================================================
elif selected_page == "3. Exploratory Data Analysis":
    render_page_header(
        title="3. Exploratory Data Analysis (EDA) & Feature Engineering",
        subtitle=(
            "Visualisasi proporsi target Exited, analisis variabel kategorikal (disertai perbandingan jumlah absolut vs "
            "churn rate %), boxplot variabel kontinu, One-Hot Encoding, dan korelasi sebelum/sesudah Feature Engineering."
        ),
        badge="Tahap 3 • Eksplorasi Data & Rekayasa Fitur",
    )

    # 1. Pie chart proporsi Churn vs Retained
    render_section_title("1. Proporsi Nasabah Churn (Exited = 1) vs Retained (Exited = 0)")
    col_pie, col_pie_stat = st.columns([1.3, 1])
    with col_pie:
        st.plotly_chart(plot_churn_pie(df_prep), use_container_width=True)
    with col_pie_stat:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>📌 Ketidakseimbangan Kelas Target (Class Imbalance)</h4>
                <p>
                    • <b>Nasabah Bertahan / Retained (Exited = 0):</b> <b>{kpi_summary['retained_count']:,} nasabah ({kpi_summary['retained_rate']:.2f}%)</b><br>
                    • <b>Nasabah Menutup Rekening / Churn (Exited = 1):</b> <b>{kpi_summary['churn_count']:,} nasabah ({kpi_summary['churn_rate']:.2f}%)</b>
                </p>
                <p style="margin-top:0.65rem;">
                    Distribusi target menunjukkan ketimpangan kelas sekitar <b>4 : 1</b> (79,63% vs 20,37%). Ketimpangan ini penting
                    diperhatikan karena model klasifikasi standar cenderung memprediksi kelas mayoritas (<i>Retained</i>), sehingga
                    diperlukan teknik penyeimbangan data latih seperti <b>SMOTE</b> pada tahap pemodelan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_insight_box(
        f"Dari <b>{kpi_summary['total_customers']:,}</b> total nasabah, sebanyak <b>{kpi_summary['churn_count']:,} nasabah ({kpi_summary['churn_rate']:.2f}%)</b> "
        f"mengalami churn, sedangkan <b>{kpi_summary['retained_count']:,} nasabah ({kpi_summary['retained_rate']:.2f}%)</b> tetap bertahan."
    )

    # 2. Categorical variables: Gender & HasCrCard per status Exited + Churn Rate (%)
    render_section_title("2. Analisis Variabel Kategorikal (Gender & HasCrCard) Beserta Churn Rate (%) per Kategori")
    cat_c1, cat_c2 = st.columns(2)

    # Statistik aktual Gender
    gen_group = df_prep.groupby("Gender")["Exited"].agg(["count", "sum", "mean"]).reset_index()
    gen_group["Retained"] = gen_group["count"] - gen_group["sum"]
    gen_group["Churn Rate (%)"] = (gen_group["mean"] * 100).round(2)
    f_row = gen_group[gen_group["Gender"] == "Female"].iloc[0]
    m_row = gen_group[gen_group["Gender"] == "Male"].iloc[0]

    # Statistik aktual HasCrCard
    cr_group = df_prep.groupby("HasCrCard")["Exited"].agg(["count", "sum", "mean"]).reset_index()
    cr_group["Retained"] = cr_group["count"] - cr_group["sum"]
    cr_group["Churn Rate (%)"] = (cr_group["mean"] * 100).round(2)
    c0_row = cr_group[cr_group["HasCrCard"] == 0].iloc[0]
    c1_row = cr_group[cr_group["HasCrCard"] == 1].iloc[0]

    with cat_c1:
        st.plotly_chart(plot_categorical_countplot(df_prep, "Gender", "Jenis Kelamin (Gender)"), use_container_width=True)
        gen_display = gen_group.rename(
            columns={"count": "Total Nasabah", "Retained": "Jumlah Retained (0)", "sum": "Jumlah Churn (1)"}
        )[["Gender", "Total Nasabah", "Jumlah Retained (0)", "Jumlah Churn (1)", "Churn Rate (%)"]]
        st.dataframe(gen_display, use_container_width=True, hide_index=True)
        render_insight_box(
            f"Secara jumlah populasi, nasabah laki-laki lebih banyak (<b>{int(m_row['count']):,}</b> vs <b>{int(f_row['count']):,}</b> perempuan). "
            f"Namun baik secara jumlah absolut maupun proporsi, nasabah <b>Perempuan (Female)</b> mencatat churn lebih tinggi yaitu "
            f"<b>{int(f_row['sum']):,} nasabah</b> dengan <b>Churn Rate {f_row['Churn Rate (%)']:.2f}%</b>, dibandingkan nasabah "
            f"<b>Laki-laki (Male)</b> sebanyak <b>{int(m_row['sum']):,} nasabah</b> dengan <b>Churn Rate hanya {m_row['Churn Rate (%)']:.2f}%</b> "
            f"(selisih risiko <b>{f_row['Churn Rate (%)'] - m_row['Churn Rate (%)']:.2f} poin persentase</b>)."
        )

    with cat_c2:
        st.plotly_chart(plot_categorical_countplot(df_prep, "HasCrCard", "Kepemilikan Kartu Kredit (HasCrCard)"), use_container_width=True)
        cr_display = cr_group.copy()
        cr_display["Kategori"] = cr_display["HasCrCard"].map({0: "Tidak Punya Kartu Kredit (0)", 1: "Punya Kartu Kredit (1)"})
        cr_display = cr_display.rename(
            columns={"count": "Total Nasabah", "Retained": "Jumlah Retained (0)", "sum": "Jumlah Churn (1)"}
        )[["Kategori", "Total Nasabah", "Jumlah Retained (0)", "Jumlah Churn (1)", "Churn Rate (%)"]]
        st.dataframe(cr_display, use_container_width=True, hide_index=True)
        render_insight_box(
            f"<b>Membedakan Jumlah Absolut vs Proporsi:</b> Secara jumlah mentah, nasabah churn yang memiliki kartu kredit terlihat jauh lebih banyak "
            f"(<b>{int(c1_row['sum']):,} nasabah</b> vs <b>{int(c0_row['sum']):,} nasabah</b> tanpa kartu kredit). Hal ini terjadi semata-mata karena mayoritas "
            f"populasi nasabah (<b>{int(c1_row['count']):,} orang / 70,55%</b>) memang memiliki kartu kredit. Jika dilihat dari proporsinya, "
            f"<b>Churn Rate pemilik kartu kredit adalah {c1_row['Churn Rate (%)']:.2f}%</b>, hampir sama dengan yang tidak memiliki kartu kredit yaitu "
            f"<b>{c0_row['Churn Rate (%)']:.2f}%</b>. Dengan demikian, <code>HasCrCard</code> bukan faktor pembeda churn yang kuat."
        )

    # 3. Continuous attributes: Boxplot CreditScore, Age, EstimatedSalary per Exited
    render_section_title("3. Analisis Variabel Kontinu (Boxplot CreditScore, Age, EstimatedSalary per Status Exited)")
    box_c1, box_c2, box_c3 = st.columns(3)

    med_cs_0 = float(df_prep[df_prep["Exited"] == 0]["CreditScore"].median())
    med_cs_1 = float(df_prep[df_prep["Exited"] == 1]["CreditScore"].median())
    mean_cs_0 = float(df_prep[df_prep["Exited"] == 0]["CreditScore"].mean())
    mean_cs_1 = float(df_prep[df_prep["Exited"] == 1]["CreditScore"].mean())

    med_age_0 = float(df_prep[df_prep["Exited"] == 0]["Age"].median())
    med_age_1 = float(df_prep[df_prep["Exited"] == 1]["Age"].median())
    mean_age_0 = float(df_prep[df_prep["Exited"] == 0]["Age"].mean())
    mean_age_1 = float(df_prep[df_prep["Exited"] == 1]["Age"].mean())

    med_sal_0 = float(df_prep[df_prep["Exited"] == 0]["EstimatedSalary"].median())
    med_sal_1 = float(df_prep[df_prep["Exited"] == 1]["EstimatedSalary"].median())
    mean_sal_0 = float(df_prep[df_prep["Exited"] == 0]["EstimatedSalary"].mean())
    mean_sal_1 = float(df_prep[df_prep["Exited"] == 1]["EstimatedSalary"].mean())

    with box_c1:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "CreditScore", "Credit Score"), use_container_width=True)
        render_insight_box(
            f"Distribusi <b>CreditScore</b> kedua kelompok cukup berdekatan: median nasabah Churn adalah <b>{med_cs_1:.0f}</b> "
            f"(rata-rata <b>{mean_cs_1:.2f}</b>), sedikit lebih rendah dibandingkan nasabah Retained dengan median <b>{med_cs_0:.0f}</b> "
            f"(rata-rata <b>{mean_cs_0:.2f}</b>), namun terdapat pencilan (<i>outliers</i>) skor kredit di bawah 400 pada kelompok Churn."
        )
    with box_c2:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "Age", "Usia Nasabah (Age)"), use_container_width=True)
        render_insight_box(
            f"Variabel <b>Age</b> memperlihatkan pemisahan distribusi paling tajam: nasabah yang Churn memiliki median usia "
            f"<b>{med_age_1:.0f} tahun</b> (rata-rata <b>{mean_age_1:.2f} tahun</b>), jauh lebih tua dibandingkan nasabah Retained "
            f"yang memiliki median <b>{med_age_0:.0f} tahun</b> (rata-rata <b>{mean_age_0:.2f} tahun</b>)."
        )
    with box_c3:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "EstimatedSalary", "Perkiraan Gaji (EstimatedSalary)"), use_container_width=True)
        render_insight_box(
            f"Kotak distribusi kuartil <b>EstimatedSalary</b> hampir sejajar antara kedua kelas: median gaji nasabah Churn sebesar "
            f"<b>${med_sal_1:,.2f}</b> (rata-rata <b>${mean_sal_1:,.2f}</b>) dan nasabah Retained sebesar <b>${med_sal_0:,.2f}</b> "
            f"(rata-rata <b>${mean_sal_0:,.2f}</b>), sehingga gaji secara tunggal bukan pemicu utama churn."
        )

    # 4. Encoding & Correlation Matrix SEBELUM Feature Engineering
    render_section_title("4. One-Hot Encoding & Heatmap Korelasi SEBELUM Feature Engineering")
    enc_c1, enc_c2 = st.columns([1, 1.35])
    with enc_c1:
        st.markdown("**Hasil `pd.get_dummies(df, drop_first=True)` (Muncul Kolom `Gender_Male`):**")
        st.dataframe(df_encoded.head(8), use_container_width=True)
        st.caption(
            "Variabel kategorikal `Gender` dikonversi menjadi kolom numerik biner `Gender_Male` (1 = Male, 0 = Female) "
            "agar dapat dihitung korelasinya dan diproses oleh model Machine Learning."
        )
    with enc_c2:
        corr_before = df_encoded.corr(numeric_only=True)
        st.plotly_chart(
            plot_correlation_heatmap(corr_before, "Correlation Matrix SEBELUM Feature Engineering"),
            use_container_width=True,
        )

    corr_age_exited = float(corr_before.loc["Age", "Exited"])
    corr_gender_exited = float(corr_before.loc["Gender_Male", "Exited"])
    corr_cs_exited = float(corr_before.loc["CreditScore", "Exited"])
    corr_card_exited = float(corr_before.loc["HasCrCard", "Exited"])
    corr_sal_exited = float(corr_before.loc["EstimatedSalary", "Exited"])
    render_insight_box(
        f"Sebelum rekayasa fitur, atribut dengan korelasi tertinggi terhadap <b>Exited</b> adalah <b>Age</b> sebesar "
        f"<b>r = {corr_age_exited:.4f}</b> (korelasi positif: semakin bertambah usia, risiko churn meningkat) dan "
        f"<b>Gender_Male</b> sebesar <b>r = {corr_gender_exited:.4f}</b> (korelasi negatif: nasabah laki-laki lebih jarang churn dibanding perempuan). "
        f"Sebaliknya, <b>CreditScore</b> (<b>r = {corr_cs_exited:.4f}</b>), <b>EstimatedSalary</b> (<b>r = {corr_sal_exited:.4f}</b>), "
        f"dan <b>HasCrCard</b> (<b>r = {corr_card_exited:.4f}</b>) memiliki korelasi linier langsung yang sangat mendekati nol."
    )

    # 5. Feature Engineering & Correlation Matrix SESUDAH Feature Engineering
    render_section_title("5. Feature Engineering (`Credit Score Given Age`) & Heatmap Korelasi SESUDAH Feature Engineering")
    fe_c1, fe_c2 = st.columns([1, 1.35])
    with fe_c1:
        st.markdown("**Penambahan Fitur Baru:** `Credit Score Given Age = CreditScore / Age`")
        st.dataframe(df_fe.head(8), use_container_width=True)
        mean_csga_0 = float(df_fe[df_fe["Exited"] == 0]["Credit Score Given Age"].mean())
        mean_csga_1 = float(df_fe[df_fe["Exited"] == 1]["Credit Score Given Age"].mean())
        st.markdown(
            f"""
            <div class="info-card" style="margin-top:0.5rem;">
                <h4>🔬 Makna Bisnis Fitur <code>Credit Score Given Age</code></h4>
                <p>
                    Fitur ini mengukur rasio skor kredit relatif terhadap usia nasabah.<br>
                    • Rata-rata pada kelompok <b>Retained (0)</b>: <b>{mean_csga_0:.2f}</b><br>
                    • Rata-rata pada kelompok <b>Churn (1)</b>: <b>{mean_csga_1:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fe_c2:
        corr_after = df_fe.corr(numeric_only=True)
        st.plotly_chart(
            plot_correlation_heatmap(corr_after, "Correlation Matrix SESUDAH Feature Engineering"),
            use_container_width=True,
        )

    corr_csga_exited = float(corr_after.loc["Credit Score Given Age", "Exited"])
    corr_csga_age = float(corr_after.loc["Credit Score Given Age", "Age"])
    render_insight_box(
        f"Sesudah feature engineering, fitur baru <b>Credit Score Given Age</b> menghasilkan korelasi negatif yang signifikan terhadap "
        f"<b>Exited</b> sebesar <b>r = {corr_csga_exited:.4f}</b> (meningkat pesat dibanding <i>CreditScore</i> murni yang hanya "
        f"<b>r = {corr_cs_exited:.4f}</b>) serta berkorelasi negatif kuat dengan <b>Age</b> (<b>r = {corr_csga_age:.4f}</b>). "
        f"Ini menunjukkan bahwa semakin rendah rasio skor kredit terhadap usia nasabah, semakin tinggi kecenderungan nasabah untuk churn."
    )


# ===========================================================================
# HALAMAN 4: MODELLING & EVALUASI
# ===========================================================================
elif selected_page == "4. Modelling & Evaluasi":
    render_page_header(
        title="4. Modelling & Evaluasi",
        subtitle=(
            "Pelatihan 4 algoritma klasifikasi dengan train_test_split (75:25, stratify=y), perbandingan skenario "
            "SEBELUM vs SESUDAH SMOTE pada data test, kurva ROC gabungan, serta eksperimen rasio SMOTE 1:1 vs 60:40."
        ),
        badge="Tahap 4 • Pemodelan & Evaluasi Model",
    )

    split_info = metrics_payload["split_info"]
    best_info = metrics_payload["best_overall"]

    # cari model dengan Recall & F1 churn tertinggi pada skenario SMOTE 1:1
    best_recall_model = max(
        metrics_payload["after_smote_1_1"].items(),
        key=lambda item: item[1]["recall_churn"],
    )

    # 1. Ringkasan Split Data
    render_section_title("1. Pembagian Data (Train-Test Split) & Pencegahan Data Leakage pada SMOTE")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card(
            "Ukuran Data Train (75%)",
            f"{split_info['train_rows']:,} Baris",
            f"Retained: {split_info['train_class_dist_before']['0']:,} | Churn: {split_info['train_class_dist_before']['1']:,}",
            "navy",
        )
    with s2:
        render_kpi_card(
            "Ukuran Data Test (25%)",
            f"{split_info['test_rows']:,} Baris",
            f"Retained: {split_info['test_class_dist']['0']:,} | Churn: {split_info['test_class_dist']['1']:,}",
            "teal",
        )
    with s3:
        render_kpi_card(
            "Terbaik (ROC AUC Tertinggi)",
            best_info["model_name"],
            f"{best_info['scenario']} (AUC: {best_info['roc_auc']:.4f})",
            "gold",
        )
    with s4:
        render_kpi_card(
            "Terbaik (Recall & F1 Churn)",
            best_recall_model[0],
            f"Sesudah SMOTE (Recall: {best_recall_model[1]['recall_churn']*100:.2f}% | F1: {best_recall_model[1]['f1_churn']:.4f})",
            "danger",
        )

    st.markdown(
        f"""
        <div class="info-card">
            <h4>🛡️ Konfigurasi Split & Aturan Bebas Data Leakage</h4>
            <p>
                • Variabel prediktor <b>X</b> mencakup <b>{split_info['num_features']} fitur</b>:
                <code>{', '.join(split_info['feature_columns'])}</code>, sedangkan target <b>y</b> adalah <code>Exited</code>.<br>
                • Pembagian dilakukan dengan <code>train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)</code>.<br>
                • <b>SMOTE hanya diterapkan pada data train (<code>X_train, y_train</code>) setelah proses split</b>. Data uji
                (<code>X_test, y_test</code> = {split_info['test_rows']:,} baris) sama sekali tidak disentuh oleh SMOTE, dan seluruh metrik
                (<i>Classification Report, Confusion Matrix,</i> serta <i>ROC AUC</i> melalui <code>predict_proba(X_test)</code>)
                dihitung murni pada <b>DATA TEST</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Selectbox Rasio SMOTE (1:1 vs sekitar 60:40)
    render_section_title("2. Pilihan Rasio SMOTE & Evaluasi Detail Tiap Model pada Data Test")
    smote_ratio_choice = st.selectbox(
        "Pilih Skenario Rasio SMOTE untuk Dibandingkan dengan Kondisi Sebelum SMOTE:",
        options=[
            "SMOTE Rasio 1:1 (Seimbang 50:50 — sampling_strategy = 1.0)",
            "SMOTE Rasio ~60:40 (Mayoritas Tetap Dominan — sampling_strategy = 0.6667)",
            "Tampilkan Ketiga Skenario Sekaligus (Sebelum SMOTE vs SMOTE 1:1 vs SMOTE 60:40)",
        ],
        index=0,
    )

    active_smote_key = (
        "after_smote_60_40"
        if "60:40" in smote_ratio_choice and "Ketiga" not in smote_ratio_choice
        else "after_smote_1_1"
    )
    active_smote_label = (
        "Sesudah SMOTE (60:40)"
        if active_smote_key == "after_smote_60_40"
        else "Sesudah SMOTE (1:1)"
    )

    tab_scen_a, tab_scen_b, tab_scen_c = st.tabs(
        [
            "🅰️ Skenario A: SEBELUM SMOTE",
            "🅱️ Skenario B: SESUDAH SMOTE (Rasio 1:1 / 50:50)",
            "🆎 Skenario Eksperimen: SESUDAH SMOTE (Rasio 60:40)",
        ]
    )

    def _render_scenario_details(scen_dict: dict, scen_label: str) -> None:
        model_names = list(scen_dict.keys())
        sub_tabs = st.tabs([f"🤖 {m}" for m in model_names])
        for idx, m_name in enumerate(model_names):
            with sub_tabs[idx]:
                m_data = scen_dict[m_name]
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    render_kpi_card("Accuracy", f"{m_data['accuracy']*100:.2f}%", f"Skor: {m_data['accuracy']:.4f}", "navy")
                with k2:
                    render_kpi_card("Precision (Churn)", f"{m_data['precision_churn']*100:.2f}%", f"Skor: {m_data['precision_churn']:.4f}", "blue")
                with k3:
                    render_kpi_card("Recall (Churn)", f"{m_data['recall_churn']*100:.2f}%", f"Skor: {m_data['recall_churn']:.4f}", "danger")
                with k4:
                    render_kpi_card("F1-Score (Churn)", f"{m_data['f1_churn']*100:.2f}%", f"Skor: {m_data['f1_churn']:.4f}", "gold")
                with k5:
                    render_kpi_card("ROC AUC (X_test)", f"{m_data['roc_auc']:.4f}", "predict_proba pada X_test", "teal")

                c_rep, c_cm = st.columns([1.25, 1])
                with c_rep:
                    st.markdown(f"**Classification Report — {m_name} ({scen_label}):**")
                    rep_df = format_classification_report_df(m_data["classification_report"])
                    st.dataframe(rep_df, use_container_width=True, hide_index=True)
                with c_cm:
                    st.plotly_chart(
                        plot_confusion_matrix(m_data["confusion_matrix"], m_name, scen_label),
                        use_container_width=True,
                    )

    with tab_scen_a:
        _render_scenario_details(metrics_payload["before_smote"], "Sebelum SMOTE")
    with tab_scen_b:
        dist_11 = split_info["train_class_dist_smote_1_1"]
        st.caption(
            f"Distribusi Data Train setelah SMOTE 1:1: Kelas 0 (Retained) = **{dist_11['0']:,}** baris (50,0%), "
            f"Kelas 1 (Churn) = **{dist_11['1']:,}** baris (50,0%)."
        )
        _render_scenario_details(metrics_payload["after_smote_1_1"], "Sesudah SMOTE 1:1")
    with tab_scen_c:
        dist_60 = split_info["train_class_dist_smote_60_40"]
        tot_60 = dist_60["0"] + dist_60["1"]
        st.caption(
            f"Distribusi Data Train setelah SMOTE 60:40: Kelas 0 (Retained) = **{dist_60['0']:,}** baris "
            f"({dist_60['0']/tot_60*100:.1f}%), Kelas 1 (Churn) = **{dist_60['1']:,}** baris ({dist_60['1']/tot_60*100:.1f}%)."
        )
        _render_scenario_details(metrics_payload["after_smote_60_40"], "Sesudah SMOTE 60:40")

    # 3. Kurva ROC Gabungan Keempat Model
    render_section_title("3. Kurva ROC Gabungan Keempat Model (Pada Data Test)")
    if "Ketiga" in smote_ratio_choice:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.plotly_chart(plot_combined_roc_curve(metrics_payload["before_smote"], "Sebelum SMOTE"), use_container_width=True)
        with rc2:
            st.plotly_chart(plot_combined_roc_curve(metrics_payload["after_smote_1_1"], "Sesudah SMOTE (1:1)"), use_container_width=True)
        with rc3:
            st.plotly_chart(plot_combined_roc_curve(metrics_payload["after_smote_60_40"], "Sesudah SMOTE (60:40)"), use_container_width=True)
    else:
        roc_c1, roc_c2 = st.columns(2)
        with roc_c1:
            st.plotly_chart(
                plot_combined_roc_curve(metrics_payload["before_smote"], "Sebelum SMOTE"),
                use_container_width=True,
            )
        with roc_c2:
            st.plotly_chart(
                plot_combined_roc_curve(metrics_payload[active_smote_key], active_smote_label),
                use_container_width=True,
            )

    gb_auc_b = metrics_payload["before_smote"]["Gradient Boosting"]["roc_auc"]
    gb_auc_11 = metrics_payload["after_smote_1_1"]["Gradient Boosting"]["roc_auc"]
    gb_auc_60 = metrics_payload["after_smote_60_40"]["Gradient Boosting"]["roc_auc"]
    rf_auc_b = metrics_payload["before_smote"]["Random Forest"]["roc_auc"]
    rf_auc_11 = metrics_payload["after_smote_1_1"]["Random Forest"]["roc_auc"]
    rf_auc_60 = metrics_payload["after_smote_60_40"]["Random Forest"]["roc_auc"]
    render_insight_box(
        f"Pada seluruh skenario pengujian terhadap <code>X_test</code>, <b>Gradient Boosting</b> konsisten memperoleh "
        f"nilai ROC AUC tertinggi (<b>{gb_auc_b:.4f}</b> sebelum SMOTE, <b>{gb_auc_11:.4f}</b> pada SMOTE 1:1, dan "
        f"<b>{gb_auc_60:.4f}</b> pada SMOTE 60:40), diikuti oleh <b>Random Forest</b> (<b>{rf_auc_b:.4f}</b> sebelum SMOTE, "
        f"<b>{rf_auc_11:.4f}</b> pada SMOTE 1:1, dan <b>{rf_auc_60:.4f}</b> pada SMOTE 60:40)."
    )

    # 4. Tabel Perbandingan Ringkas & Penandaan Model Terbaik + Trade-off
    render_section_title("4. Tabel Perbandingan Model & Analisis Trade-off Recall vs Accuracy")
    comp_df = build_comparison_dataframe(metrics_payload, include_60_40=True)
    if "Ketiga" not in smote_ratio_choice:
        comp_df = comp_df[comp_df["Skenario"].isin(["Sebelum SMOTE", active_smote_label])].reset_index(drop=True)

    max_auc_val = comp_df["ROC AUC"].max()
    max_rec_val = comp_df["Recall (Churn)"].max()
    max_f1_val = comp_df["F1-Score (Churn)"].max()

    def _badge_model(row: pd.Series) -> str:
        badges = []
        if row["ROC AUC"] == max_auc_val:
            badges.append("🏆 ROC AUC Terbaik")
        if row["Recall (Churn)"] == max_rec_val:
            badges.append("🎯 Recall Churn Tertinggi")
        if row["F1-Score (Churn)"] == max_f1_val:
            badges.append("⭐ F1 Churn Tertinggi")
        return " | ".join(badges) if badges else "Kandidat"

    comp_df["Penandaan Model Terbaik"] = comp_df.apply(_badge_model, axis=1)

    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC AUC"],
            color="#D1FAE5",
        ),
        use_container_width=True,
        hide_index=True,
    )

    gb_b = metrics_payload["before_smote"]["Gradient Boosting"]
    gb_11 = metrics_payload["after_smote_1_1"]["Gradient Boosting"]
    gb_60 = metrics_payload["after_smote_60_40"]["Gradient Boosting"]
    rf_b = metrics_payload["before_smote"]["Random Forest"]
    rf_11 = metrics_payload["after_smote_1_1"]["Random Forest"]
    rf_60 = metrics_payload["after_smote_60_40"]["Random Forest"]

    st.markdown(
        f"""
        <div class="info-card" style="border-left: 5px solid {COLORS['navy']}; margin-top: 0.8rem;">
            <h4>⚖️ Penjelasan Trade-Off: Mengapa SMOTE Menaikkan Recall Churn tetapi Menurunkan Accuracy?</h4>
            <p>
                1. <b>Kondisi Sebelum SMOTE (Akurasi Tinggi namun Recall Rendah):</b> Karena 79,63% data adalah nasabah bertahan (<i>Retained</i>),
                model yang jarang menebak <i>Churn</i> secara otomatis terlihat memiliki <b>Accuracy tinggi</b> (misalnya <b>Gradient Boosting {gb_b['accuracy']*100:.2f}%</b>
                dan <b>Random Forest {rf_b['accuracy']*100:.2f}%</b>). Namun, <b>Recall kelas Churn sangat rendah ({gb_b['recall_churn']*100:.2f}% pada Gradient Boosting
                dan {rf_b['recall_churn']*100:.2f}% pada Random Forest)</b> — artinya mayoritas nasabah yang benar-benar akan menutup rekening lolos tanpa terdeteksi.<br><br>
                2. <b>Kondisi Sesudah SMOTE 1:1 (Recall &amp; F1 Melonjak, Accuracy Turun):</b> Setelah kelas minoritas diseimbangkan pada data train,
                model menjadi jauh lebih sensitif mengenali pola nasabah churn. Pada <b>Gradient Boosting</b>, <b>Recall Churn melonjak lebih dari 2x lipat menjadi
                {gb_11['recall_churn']*100:.2f}%</b> (F1-Score naik dari {gb_b['f1_churn']:.4f} ke <b>{gb_11['f1_churn']:.4f}</b>), dan pada <b>Random Forest</b>
                Recall naik menjadi <b>{rf_11['recall_churn']*100:.2f}%</b>. Konsekuensinya, sebagian nasabah bertahan ikut terprediksi berisiko churn
                (<i>False Positive</i> bertambah), sehingga <b>Accuracy turun menjadi {gb_11['accuracy']*100:.2f}%</b> (Gradient Boosting) dan
                <b>{rf_11['accuracy']*100:.2f}%</b> (Random Forest).<br><br>
                3. <b>Keunggulan Rasio SMOTE 60:40:</b> Dengan menjaga kelas mayoritas tetap dominan (60% : 40%), model mencapai titik kompromi optimal:
                pada <b>Gradient Boosting</b>, <b>F1-Score mencapai titik tertinggi ({gb_60['f1_churn']:.4f})</b> dengan <b>Accuracy kembali naik ke {gb_60['accuracy']*100:.2f}%</b>,
                <b>ROC AUC {gb_60['roc_auc']:.4f}</b>, dan <b>Recall Churn tetap kuat di {gb_60['recall_churn']*100:.2f}%</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HALAMAN 5: FEATURE IMPORTANCE
# ===========================================================================
elif selected_page == "5. Feature Importance":
    render_page_header(
        title="5. Analisis Feature Importance (Random Forest)",
        subtitle=(
            "Mengukur tingkat kontribusi setiap fitur dalam memisahkan kelas Churn vs Retained pada algoritma "
            "Random Forest serta narasi faktor paling berpengaruh berdasarkan hasil aktual."
        ),
        badge="Tahap 5 • Tingkat Kepentingan Fitur",
    )

    fi_scen_choice = st.radio(
        "Pilih Skenario Pelatihan Random Forest:",
        options=["Sebelum SMOTE", "Sesudah SMOTE (1:1)", "Sesudah SMOTE (60:40)"],
        horizontal=True,
    )

    scen_map_fi = {
        "Sebelum SMOTE": metrics_payload["before_smote"]["Random Forest"]["feature_importances"],
        "Sesudah SMOTE (1:1)": metrics_payload["after_smote_1_1"]["Random Forest"]["feature_importances"],
        "Sesudah SMOTE (60:40)": metrics_payload["after_smote_60_40"]["Random Forest"]["feature_importances"],
    }
    fi_data = scen_map_fi[fi_scen_choice]
    sorted_fi = sorted(fi_data.items(), key=lambda x: x[1], reverse=True)

    render_section_title(f"1. Grafik Horizontal Feature Importance — Random Forest ({fi_scen_choice})")
    fi_col1, fi_col2 = st.columns([1.35, 1])
    with fi_col1:
        st.plotly_chart(
            plot_feature_importance(fi_data, f"Feature Importance Random Forest ({fi_scen_choice})"),
            use_container_width=True,
        )
    with fi_col2:
        fi_table = pd.DataFrame(
            [
                {
                    "Peringkat": i + 1,
                    "Nama Fitur": k,
                    "Bobot Kepentingan": round(v, 4),
                    "Persentase Kontribusi (%)": f"{v*100:.2f}%",
                }
                for i, (k, v) in enumerate(sorted_fi)
            ]
        )
        st.markdown("**Tabel Peringkat Kontribusi Fitur Aktual:**")
        st.dataframe(fi_table, use_container_width=True, hide_index=True)

    # Bandingkan ketiga skenario berdampingan
    render_section_title("2. Perbandingan Feature Importance di Ketiga Skenario")
    fi_comp_rows = []
    for feat in metrics_payload["split_info"]["feature_columns"]:
        fi_comp_rows.append(
            {
                "Nama Fitur": feat,
                "Sebelum SMOTE (%)": round(metrics_payload["before_smote"]["Random Forest"]["feature_importances"][feat] * 100, 2),
                "Sesudah SMOTE 1:1 (%)": round(metrics_payload["after_smote_1_1"]["Random Forest"]["feature_importances"][feat] * 100, 2),
                "Sesudah SMOTE 60:40 (%)": round(metrics_payload["after_smote_60_40"]["Random Forest"]["feature_importances"][feat] * 100, 2),
            }
        )
    df_fi_comp = pd.DataFrame(fi_comp_rows).sort_values("Sesudah SMOTE 1:1 (%)", ascending=False)
    st.dataframe(df_fi_comp, use_container_width=True, hide_index=True)

    top1_name, top1_val = sorted_fi[0]
    top2_name, top2_val = sorted_fi[1]
    top3_name, top3_val = sorted_fi[2]
    top4_name, top4_val = sorted_fi[3]

    render_section_title("3. Narasi Faktor yang Paling Berpengaruh Berdasarkan Hasil Aktual")
    st.markdown(
        f"""
        <div class="info-card" style="border-left: 5px solid {COLORS['teal']};">
            <h4>📌 Interpretasi Hasil Aktual Feature Importance vs Korelasi EDA</h4>
            <p>
                1. <b>Dominasi Fitur Kontinu &amp; Fitur Hasil Rekayasa:</b> Pada skenario <b>{fi_scen_choice}</b>, empat fitur dengan
                kontribusi terbesar adalah <b>{top1_name} ({top1_val*100:.2f}%)</b>, <b>{top2_name} ({top2_val*100:.2f}%)</b>,
                <b>{top3_name} ({top3_val*100:.2f}%)</b>, dan <b>{top4_name} ({top4_val*100:.2f}%)</b>. Keempat atribut kontinu ini
                menyumbang lebih dari <b>96%</b> total penurunan <i>Gini Impurity</i> pada pohon keputusan Random Forest.<br><br>
                2. <b>Perubahan Peringkat Sebelum vs Sesudah SMOTE:</b><br>
                • <b>Sebelum SMOTE:</b> <code>EstimatedSalary</code> (34,47%) dan <code>Credit Score Given Age</code> (26,65%) menempati urutan
                teratas karena pohon keputusan banyak melakukan pemisahan pada nilai desimal kontinu yang rapat, disusul <code>CreditScore</code> (20,76%) dan <code>Age</code> (16,13%).<br>
                • <b>Sesudah SMOTE (1:1):</b> Ketika kedua kelas diseimbangkan, <b><code>Age</code> naik menjadi fitur paling penting nomor 1 (27,63%)</b>,
                diikuti ketat oleh <b><code>Credit Score Given Age</code> (25,79%)</b>, <code>EstimatedSalary</code> (24,49%), dan <code>CreditScore</code> (18,93%).<br><br>
                3. <b>Mengapa <code>Gender_Male</code> &amp; <code>HasCrCard</code> Bobotnya Kecil di Pohon Keputusan?</b><br>
                Variabel biner (0/1) hanya memiliki satu titik potong (<i>split point</i>), berbeda dengan variabel kontinu yang memiliki ratusan
                titik potong di sepanjang cabang pohon. Namun, sebagaimana dibuktikan pada tahap EDA, <b>Usia (Age)</b>, rasio
                <b>Credit Score Given Age</b>, dan <b>Gender (Perempuan)</b> merupakan faktor pembeda risiko churn paling nyata secara arah bisnis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HALAMAN 6: PREDIKSI NASABAH
# ===========================================================================
elif selected_page == "6. Prediksi Nasabah":
    render_page_header(
        title="6. Simulasi Prediksi Risiko Churn Nasabah",
        subtitle=(
            "Masukkan 5 atribut dasar nasabah (Gender, Age, CreditScore, EstimatedSalary, HasCrCard). "
            "Fitur Credit Score Given Age dihitung otomatis untuk memprediksi probabilitas churn beserta rekomendasi retensi."
        ),
        badge="Tahap 6 • Kalkulator Prediksi Real-Time",
    )

    col_form, col_result = st.columns([1.05, 1.25])

    with col_form:
        render_section_title("Formulir Input Profil Nasabah (5 Fitur Dasar)")
        with st.form("churn_prediction_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                gender_in = st.selectbox(
                    "1. Jenis Kelamin (Gender)",
                    options=["Female", "Male"],
                    format_func=lambda g: "Perempuan (Female)" if g == "Female" else "Laki-laki (Male)",
                )
                age_in = st.number_input(
                    "2. Usia Nasabah (Age — Tahun)",
                    min_value=18,
                    max_value=95,
                    value=46,
                    step=1,
                )
                has_card_in = st.selectbox(
                    "3. Kepemilikan Kartu Kredit (HasCrCard)",
                    options=[1, 0],
                    format_func=lambda v: "1 — Memiliki Kartu Kredit" if v == 1 else "0 — Tidak Memiliki Kartu Kredit",
                )
            with f_c2:
                credit_score_in = st.number_input(
                    "4. Skor Kredit (CreditScore: 300 – 850)",
                    min_value=300,
                    max_value=850,
                    value=580,
                    step=5,
                )
                salary_in = st.number_input(
                    "5. Perkiraan Gaji (EstimatedSalary — $)",
                    min_value=0.0,
                    max_value=500000.0,
                    value=101348.88,
                    step=1000.0,
                    format="%.2f",
                )
                rf_mode_in = st.selectbox(
                    "Varian Model Random Forest",
                    options=[
                        "Sesudah SMOTE (60:40)",
                        "Sesudah SMOTE (1:1)",
                        "Sebelum SMOTE",
                    ],
                    index=0,
                    help="Pilih skenario model Random Forest yang digunakan untuk menghitung probabilitas churn.",
                )

            live_csga = float(credit_score_in) / float(age_in)
            st.info(
                f"⚙️ **Fitur Rekayasa Otomatis (`Credit Score Given Age`):** "
                f"`{credit_score_in} / {age_in}` = **{live_csga:.4f}**"
            )

            st.form_submit_button(
                "🔍 Hitung Probabilitas Churn Nasabah",
                use_container_width=True,
                type="primary",
            )

    try:
        pred_out = predict_customer_churn(
            pipeline_bundle=pipeline_bundle,
            gender=gender_in,
            age=int(age_in),
            credit_score=int(credit_score_in),
            estimated_salary=float(salary_in),
            has_cr_card=int(has_card_in),
            rf_mode=rf_mode_in,
        )
    except ValueError as err:
        st.error(f"Input tidak valid: {err}")
        st.stop()

    with col_result:
        render_section_title("Hasil Prediksi & Kategori Risiko Churn")
        st.plotly_chart(plot_churn_probability_gauge(pred_out["prob_churn"]), use_container_width=True)

        r_c1, r_c2, r_c3 = st.columns(3)
        accent_risk = (
            "danger"
            if pred_out["risk_category"] == "Tinggi"
            else "gold"
            if pred_out["risk_category"] == "Sedang"
            else "teal"
        )
        with r_c1:
            render_kpi_card(
                "Probabilitas Churn",
                f"{pred_out['prob_churn']*100:.2f}%",
                f"Probabilitas Bertahan: {pred_out['prob_retained']*100:.2f}%",
                accent_risk,
            )
        with r_c2:
            render_kpi_card(
                "Kategori Risiko",
                f"Risiko {pred_out['risk_category']}",
                "Rendah <35% | Sedang 35–65% | Tinggi >65%",
                accent_risk,
            )
        with r_c3:
            render_kpi_card(
                "Credit Score Given Age",
                f"{pred_out['credit_score_given_age']:.2f}",
                f"Otomatis ({credit_score_in} / {age_in})",
                "navy",
            )

        rec_items_html = "".join(
            f"<li style='margin-bottom:0.35rem;'>{item}</li>" for item in pred_out["recommendations"]
        )
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 5px solid {pred_out['risk_color']};">
                <h4>📋 Saran Tindakan Retensi (Kategori Risiko {pred_out['risk_category']})</h4>
                <ul style="margin: 0.4rem 0 0 1.1rem; padding: 0; font-size: 0.92rem; line-height: 1.55;">
                    {rec_items_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_title("Vektor Fitur yang Masuk ke Pipeline Model Random Forest")
    st.dataframe(pred_out["input_dataframe"], use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 7: KESIMPULAN DAN REKOMENDASI
# ===========================================================================
elif selected_page == "7. Kesimpulan & Rekomendasi":
    render_page_header(
        title="7. Kesimpulan dan Rekomendasi Bisnis",
        subtitle=(
            "Sintesis akhir yang menjawab ketiga objective bisnis berdasarkan angka aktual Dataset1_Customer_Churn.csv, "
            "rekomendasi tindakan retensi per segmen nasabah, serta keterbatasan dataset."
        ),
        badge="Tahap 7 • Kesimpulan & Rekomendasi",
    )

    corr_after = df_fe.corr(numeric_only=True)
    corr_age = float(corr_after.loc["Age", "Exited"])
    corr_csga = float(corr_after.loc["Credit Score Given Age", "Exited"])
    corr_gender = float(corr_after.loc["Gender_Male", "Exited"])

    female_rate = float(df_raw[df_raw["Gender"] == "Female"]["Exited"].mean() * 100.0)
    male_rate = float(df_raw[df_raw["Gender"] == "Male"]["Exited"].mean() * 100.0)
    card1_rate = float(df_raw[df_raw["HasCrCard"] == 1]["Exited"].mean() * 100.0)
    card0_rate = float(df_raw[df_raw["HasCrCard"] == 0]["Exited"].mean() * 100.0)

    df_age_seg = df_raw.copy()
    df_age_seg["Kelompok Usia"] = pd.cut(
        df_age_seg["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["≤ 30 Tahun", "31 – 40 Tahun", "41 – 50 Tahun", "51 – 60 Tahun", "> 60 Tahun"],
    )
    seg_summary = (
        df_age_seg.groupby("Kelompok Usia", observed=False)["Exited"]
        .agg(Total_Nasabah="count", Jumlah_Churn="sum", Churn_Rate="mean")
        .reset_index()
    )
    seg_summary["Churn Rate (%)"] = (seg_summary["Churn_Rate"] * 100).round(2)
    highest_risk_seg = seg_summary.sort_values("Churn Rate (%)", ascending=False).iloc[0]

    gb_b = metrics_payload["before_smote"]["Gradient Boosting"]
    gb_11 = metrics_payload["after_smote_1_1"]["Gradient Boosting"]
    gb_60 = metrics_payload["after_smote_60_40"]["Gradient Boosting"]
    rf_b = metrics_payload["before_smote"]["Random Forest"]
    rf_11 = metrics_payload["after_smote_1_1"]["Random Forest"]
    rf_60 = metrics_payload["after_smote_60_40"]["Random Forest"]

    render_section_title("1. Kesimpulan Analisis (Menjawab Tiga Objective Bisnis)")
    c_obj1, c_obj2, c_obj3 = st.columns(3)

    with c_obj1:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['navy']}; height: 100%;">
                <h4>🔍 Objective (a): Faktor Paling Memengaruhi Churn</h4>
                <p>
                    1. <b>Usia (Age) &amp; Rasio Kredit per Usia:</b> Faktor penentu terkuat secara korelasi dan distribusi adalah
                    <b>Age</b> (korelasi positif <b>r = {corr_age:.4f}</b>; median usia churn <b>45 tahun</b> vs retained <b>36 tahun</b>)
                    serta fitur hasil rekayasa <b>Credit Score Given Age</b> (korelasi negatif <b>r = {corr_csga:.4f}</b>). Kelompok usia
                    <b>{highest_risk_seg['Kelompok Usia']}</b> memiliki tingkat churn tertinggi hingga <b>{highest_risk_seg['Churn Rate (%)']:.2f}%</b>.<br><br>
                    2. <b>Jenis Kelamin (Gender):</b> Nasabah perempuan memiliki churn rate jauh lebih tinggi (<b>{female_rate:.2f}%</b>)
                    dibandingkan nasabah laki-laki (<b>{male_rate:.2f}%</b>) dengan korelasi <code>Gender_Male</code> sebesar <b>r = {corr_gender:.4f}</b>.<br><br>
                    3. <b>Feature Importance Random Forest:</b> Pada model pasca-SMOTE, fitur kontinu <b>Age (27,63%)</b>,
                    <b>Credit Score Given Age (25,79%)</b>, <b>EstimatedSalary (24,49%)</b>, dan <b>CreditScore (18,93%)</b> mendominasi
                    daya pisah model, sedangkan <b>HasCrCard</b> terbukti bukan pembeda kuat (churn rate pemilik kartu <b>{card1_rate:.2f}%</b> vs non-pemilik <b>{card0_rate:.2f}%</b>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_obj2:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['teal']}; height: 100%;">
                <h4>🏆 Objective (b): Model Prediksi Paling Akurat</h4>
                <p>
                    1. <b>Model Terbaik Berdasarkan ROC AUC:</b> Algoritma <b>Gradient Boosting</b> unggul di seluruh skenario
                    dengan <b>ROC AUC = {gb_b['roc_auc']:.4f}</b> (Accuracy <b>{gb_b['accuracy']*100:.2f}%</b>) sebelum SMOTE, disusul oleh
                    <b>Random Forest</b> (ROC AUC = <b>{rf_b['roc_auc']:.4f}</b>, Accuracy <b>{rf_b['accuracy']*100:.2f}%</b>).<br><br>
                    2. <b>Model Terbaik untuk Deteksi Churn (Recall &amp; F1):</b> Sebelum SMOTE, Recall churn sangat rendah
                    ({gb_b['recall_churn']*100:.2f}% pada Gradient Boosting). Sesudah <b>SMOTE 1:1</b>, Recall churn melonjak ke
                    <b>{gb_11['recall_churn']*100:.2f}%</b> (F1 = <b>{gb_11['f1_churn']:.4f}</b>, AUC = <b>{gb_11['roc_auc']:.4f}</b>).<br><br>
                    3. <b>Keseimbangan Terbaik dengan SMOTE 60:40:</b> Rasio SMOTE 60:40 menghasilkan <b>F1-Score tertinggi ({gb_60['f1_churn']:.4f})</b>
                    pada Gradient Boosting dengan <b>Accuracy {gb_60['accuracy']*100:.2f}%</b>, <b>Recall {gb_60['recall_churn']*100:.2f}%</b>, dan
                    <b>ROC AUC {gb_60['roc_auc']:.4f}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_obj3:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['gold']}; height: 100%;">
                <h4>🚀 Objective (c): Pemanfaatan untuk Pertumbuhan Bisnis</h4>
                <p>
                    1. <b>Targeted Retention Campaign:</b> Mengalokasikan anggaran retensi Divisi Marketing secara presisi pada
                    nasabah yang diprediksi berisiko tinggi (probabilitas &gt; 65%), terutama pada segmen usia <b>41–60 tahun</b>
                    dan segmen nasabah perempuan.<br><br>
                    2. <b>Optimalisasi Biaya Retensi:</b> Mengingat <code>HasCrCard</code> memiliki churn rate yang hampir identik
                    antara pemilik dan non-pemilik kartu kredit, bank tidak perlu menghamburkan promosi kartu kredit massal untuk
                    menekan churn, melainkan fokus pada produk manajemen kekayaan/pensiun bagi nasabah senior.<br><br>
                    3. <b>Peningkatan Customer Lifetime Value (CLV):</b> Menekan laju churn dari {kpi_summary['churn_rate']:.2f}%
                    secara bertahap akan menjaga dana pihak ketiga tetap stabil dan memperkuat pertumbuhan laba bank.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_title("2. Tabel Risiko Churn Berdasarkan Segmen Kelompok Usia (Data Aktual)")
    st.dataframe(
        seg_summary[["Kelompok Usia", "Total_Nasabah", "Jumlah_Churn", "Churn Rate (%)"]].rename(
            columns={"Total_Nasabah": "Total Nasabah", "Jumlah_Churn": "Jumlah Churn"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    render_section_title("3. Rekomendasi Strategis & Keterbatasan Studi")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 5px solid {COLORS['teal']}; height: 100%;">
                <h4>💡 Rekomendasi Konkret untuk Divisi Marketing</h4>
                <p>
                    • <b>Program Retensi Nasabah Usia Matang (41–60 Tahun):</b> Kelompok usia 51–60 tahun memiliki churn rate
                    <b>56,21%</b> dan usia 41–50 tahun sebesar <b>33,97%</b>. Sediakan layanan <i>priority wealth management</i>,
                    bunga deposito kompetitif, serta pendampingan <i>Relationship Manager</i> bagi segmen usia ini.<br><br>
                    • <b>Program Loyalitas Segmen Perempuan:</b> Karena churn rate nasabah perempuan mencapai <b>{female_rate:.2f}%</b>
                    (vs laki-laki <b>{male_rate:.2f}%</b>), rancang produk tabungan berhadiah, <i>cashback merchant</i> keluarga/gaya hidup,
                    dan survei kepuasan berkala khusus segmen nasabah perempuan.<br><br>
                    • <b>Pendampingan Nasabah dengan <code>Credit Score Given Age</code> Rendah:</b> Berikan edukasi finansial dan
                    fasilitas restrukturisasi kredit ringan bagi nasabah berusia lanjut yang skor kreditnya menurun.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rec_col2:
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 5px solid {COLORS['danger']}; height: 100%;">
                <h4>⚠️ Keterbatasan Dataset &amp; Saran Pengembangan</h4>
                <p>
                    • <b>Keterbatasan Jumlah Fitur (Hanya 5 Fitur Prediktor Dasar):</b> Dataset <code>Dataset1_Customer_Churn.csv</code>
                    hanya memuat 5 fitur input (<code>Gender, Age, CreditScore, EstimatedSalary, HasCrCard</code>) ditambah 1 fitur hasil
                    rekayasa (<code>Credit Score Given Age</code>). Ketiadaan fitur aktivitas transaksi atau saldo rekening menyebabkan
                    plafon ROC AUC maksimal berada di kisaran <b>0,77</b>.<br><br>
                    • <b>Ketidakseimbangan Kelas Target (79,63% vs 20,37%):</b> Penggunaan SMOTE 1:1 meningkatkan Recall secara drastis
                    namun menurunkan Precision dan Accuracy. Oleh karena itu, penggunaan <b>SMOTE rasio 60:40</b> serta penyesuaian
                    <i>probability threshold</i> sangat disarankan saat implementasi produksi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# HALAMAN 8: TENTANG (ABOUT)
# ===========================================================================
elif selected_page == "8. Tentang":
    render_page_header(
        title="8. Tentang Aplikasi & Metodologi",
        subtitle=(
            "Dokumentasi sumber dataset Dataset1_Customer_Churn.csv, tahapan metodologi analisis DigitalSkola, "
            "keterbatasan pemodelan, dan identitas pengembang."
        ),
        badge="Tahap 8 • Dokumentasi Proyek",
    )

    col_ab1, col_ab2 = st.columns([1.3, 1])

    with col_ab1:
        render_section_title("Deskripsi Dataset & Metodologi Analisis")
        st.markdown(
            f"""
            <div class="info-card">
                <h4>📂 Spesifikasi Dataset (<code>{DEFAULT_DATA_PATH}</code>)</h4>
                <p>
                    Aplikasi ini menggunakan dataset resmi <b><code>Dataset1_Customer_Churn.csv</code></b> yang terdiri dari
                    <b>{len(df_raw):,} baris</b> dan <b>{len(df_raw.columns)} kolom</b> tanpa nilai kosong (<i>0 missing values</i>)
                    dan tanpa baris duplikat (<i>0 duplicates</i>):<br>
                    • <code>CustomerId</code>: ID unik nasabah (dihapus sebelum pemodelan).<br>
                    • <code>Gender</code>: Jenis kelamin nasabah (<code>Male</code> / <code>Female</code>).<br>
                    • <code>Age</code>: Usia nasabah dalam tahun.<br>
                    • <code>CreditScore</code>: Nilai skor kredit berdasarkan perilaku transaksi.<br>
                    • <code>EstimatedSalary</code>: Perkiraan pendapatan tahunan nasabah.<br>
                    • <code>HasCrCard</code>: Status kepemilikan kartu kredit (<code>0</code> = Tidak, <code>1</code> = Ya).<br>
                    • <code>Exited</code>: Target status penutupan rekening (<code>0</code> = Retained, <code>1</code> = Churn).
                </p>
                <h4 style="margin-top: 1rem;">🔄 Ringkasan Alur Metodologi (Datamine Team, DigitalSkola)</h4>
                <p>
                    1. <b>Business Understanding:</b> Penetapan 3 objective bisnis Divisi Marketing bank.<br>
                    2. <b>Data Understanding &amp; Preparation:</b> Validasi kolom, <code>df.drop('CustomerId')</code>, pengecekan <code>info()</code>, <code>isnull().sum()</code>, dan <code>describe()</code>.<br>
                    3. <b>EDA &amp; Feature Engineering:</b> Analisis proporsi vs jumlah absolut, encoding <code>pd.get_dummies(drop_first=True)</code> menghasilkan <code>Gender_Male</code>, serta pembuatan fitur <code>Credit Score Given Age = CreditScore / Age</code>.<br>
                    4. <b>Modelling &amp; Evaluasi Bebas Data Leakage:</b> Split data 75:25 terstratifikasi (<code>random_state=42</code>), penerapan SMOTE (1:1 dan 60:40) <b>hanya pada data train</b>, dan evaluasi 4 algoritma pada <code>X_test</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Keterbatasan Studi (Limitations)")
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {COLORS['danger']};">
                <h4>⚠️ Ruang Lingkup &amp; Keterbatasan</h4>
                <p>
                    1. <b>Jumlah Fitur Terbatas (5 Fitur Prediktor):</b> Model hanya mengandalkan profil dasar nasabah
                    (<code>Gender, Age, CreditScore, EstimatedSalary, HasCrCard</code>) beserta rasio <code>Credit Score Given Age</code>,
                    sehingga variasi perilaku transaksi harian belum tertangkap sepenuhnya.<br>
                    2. <b>Distribusi Kelas Tidak Seimbang:</b> Proporsi nasabah churn adalah <b>{kpi_summary['churn_rate']:.2f}%</b>
                    ({kpi_summary['churn_count']:,} dari {kpi_summary['total_customers']:,} nasabah), sehingga pemilihan model harus
                    mempertimbangkan <i>trade-off</i> antara <b>ROC AUC</b>, <b>Recall (Churn)</b>, dan <b>Accuracy</b> sesuai kebutuhan kampanye retensi bank.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ab2:
        render_section_title("Identitas Pengembang")
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['gold']};">
                <h4>👤 Profil Penyusun</h4>
                <p>
                    • <b>Nama Lengkap:</b> <code>Afrizal Subagja</code> <span style="color:#6B7280;font-size:0.82rem;">([Nama Anda])</span><br>
                    • <b>NIM / ID Peserta:</b> <code>072926002</code> <span style="color:#6B7280;font-size:0.82rem;">([NIM])</span><br>
                    • <b>Mata Kuliah / Proyek:</b> Pembelajaran Mesin — Evaluasi Tengah Semester (ETS)<br>
                    • <b>Referensi Metodologi:</b> Final Project "Bank Customer Churn" — Datamine Team, DigitalSkola
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Stack Teknologi & Reprodusibilitas")
        st.markdown(
            """
            <div class="info-card">
                <h4>🛠️ Spesifikasi Teknis</h4>
                <p>
                    • <b>Framework Web:</b> Streamlit &amp; Streamlit Option Menu<br>
                    • <b>Olahan Data:</b> Pandas &amp; NumPy<br>
                    • <b>Machine Learning:</b> Scikit-Learn &amp; Imbalanced-Learn (SMOTE)<br>
                    • <b>Visualisasi:</b> Plotly Express &amp; Plotly Graph Objects<br>
                    • <b>Artefak Terkompresi:</b> Joblib (<code>model/rf_pipeline.pkl</code>) &amp; JSON (<code>model/metrics.json</code>)<br>
                    • <b>Reprodusibilitas:</b> <code>random_state=42</code> pada seluruh proses split, SMOTE, dan model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
