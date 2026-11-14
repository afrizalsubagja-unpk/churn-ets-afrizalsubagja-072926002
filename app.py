"""
Aplikasi Web Streamlit: Bank Customer Churn Analytics
Mereplikasi alur analisis panduan Final Project "Bank Customer Churn" (Datamine Team, DigitalSkola).
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
# PEMUATAN DATA & VALIDASI KOLOM WAJIB
# ---------------------------------------------------------------------------
df_raw, ignored_columns = get_validated_data("data/Churn_Modelling.csv")
kpi_summary = compute_kpi_summary(df_raw)

# Tahapan pemrosesan sesuai urutan panduan
df_prep = prepare_data(df_raw)
df_encoded = encode_data(df_prep)
df_fe = engineer_features(df_encoded)

# Muat model dan hasil evaluasi dari model/ atau latih otomatis jika belum ada
pipeline_bundle, metrics_payload = load_or_train_artifacts(df_raw)

# ---------------------------------------------------------------------------
# NAVIGASI SIDEBAR (8 HALAMAN SESUAI URUTAN PANDUAN)
# ---------------------------------------------------------------------------
MENU_OPTIONS = [
    "1. Business Understanding",
    "2. Data Understanding",
    "3. Data Preparation",
    "4. Exploratory Data Analysis",
    "5. Modelling & Evaluation",
    "6. Kesimpulan & Saran",
    "7. Prediksi Nasabah",
    "8. Tentang",
]

MENU_ICONS = [
    "briefcase-fill",
    "table",
    "sliders",
    "bar-chart-line-fill",
    "cpu-fill",
    "check2-circle",
    "person-check-fill",
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
                "font-size": "13.5px",
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
                📌 Ringkasan Cepat Dataset
            </div>
            <div>• Total Data: <b>{kpi_summary['total_customers']:,}</b> baris</div>
            <div>• Kolom Aktif: <b>{len(REQUIRED_COLUMNS)}</b> kolom wajib</div>
            <div>• Churn Rate: <b>{kpi_summary['churn_rate']:.2f}%</b></div>
            <div>• Model Terbaik: <b>{metrics_payload['best_overall']['model_name']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HALAMAN 1: BUSINESS UNDERSTANDING
# ===========================================================================
if selected_page == "1. Business Understanding":
    render_page_header(
        title="1. Business Understanding",
        subtitle=(
            "Memahami konteks bisnis perbankan, dampak finansial dari penutupan rekening (churn), "
            "serta penetapan tiga tujuan strategis analisis data."
        ),
        badge="Tahap 1 • Pemahaman Bisnis",
    )

    # Baris KPI Utama sesuai spesifikasi halaman awal
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
                    Berdasarkan data aktual sebanyak <b>{kpi_summary['total_customers']:,} nasabah</b>, tercatat sejumlah
                    <b style="color:{COLORS['danger']};">{kpi_summary['churn_count']:,} nasabah ({kpi_summary['churn_rate']:.2f}%)</b>
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
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {COLORS['gold']};">
                <h4>📊 Potret Awal Risiko Churn dari Data</h4>
                <p>
                    • <b>Rasio Ketidakseimbangan Kelas:</b> Sekitar <b>{kpi_summary['retained_rate']:.1f}% : {kpi_summary['churn_rate']:.1f}%</b>
                    antara nasabah bertahan dan <i>churn</i>.<br>
                    • <b>Usia Rata-rata Nasabah Churn:</b> <b>{avg_age_churn:.1f} tahun</b> (lebih tinggi dibandingkan nasabah bertahan yaitu <b>{avg_age_ret:.1f} tahun</b>).<br>
                    • <b>Kepemilikan Kartu Kredit:</b> <b>{kpi_summary['cr_card_pct']:.1f}%</b> nasabah memiliki kartu kredit.<br>
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
                    Mengidentifikasi <b>faktor atau atribut apa yang paling penting dan berpengaruh</b> dalam menyebabkan
                    tingginya tingkat <i>churn</i> nasabah melalui analisis korelasi, rekayasa fitur (<i>Feature Engineering</i>),
                    dan <i>Feature Importance</i> model.
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
                    baik <b>sebelum maupun sesudah SMOTE</b> untuk menentukan model mana yang paling akurat dan andal
                    memprediksi nasabah yang akan <i>churn</i>.
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
                    Merumuskan bagaimana perusahaan/bank dapat <b>memanfaatkan hasil analisis dan model prediktif ini</b>
                    untuk menyusun kampanye retensi yang tepat sasaran, menekan laju <i>churn</i>, dan meningkatkan
                    pertumbuhan profit bisnis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Alur Metodologi Panduan
    render_section_title("Alur Metodologi Analisis (Panduan Final Project DigitalSkola)")
    st.markdown(
        """
        <div class="info-card">
            <p>
                Aplikasi ini mereplikasi secara presisi setiap tahapan pada panduan <b>Final Project "Bank Customer Churn" (Datamine Team, DigitalSkola)</b>:
                dimulai dari <b>(1) Business Understanding</b> ➔ <b>(2) Data Understanding</b> (7 kolom wajib) ➔
                <b>(3) Data Preparation</b> (penghapusan <code>CustomerId</code>, pengecekan <i>missing value</i> & statistik deskriptif) ➔
                <b>(4) Exploratory Data Analysis</b> (visualisasi kategorikal, kontinu, <i>One-Hot Encoding</i>, dan rekayasa fitur <code>Credit Score Given Age</code>) ➔
                <b>(5) Modelling & Evaluation</b> (perbandingan 4 model sebelum & sesudah <i>SMOTE</i> pada data <i>test</i>) ➔
                <b>(6) Kesimpulan, Rekomendasi & Eksperimen Rasio SMOTE 60:40</b> ➔ hingga <b>(7) Simulasi Prediksi Nasabah Baru</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HALAMAN 2: DATA UNDERSTANDING
# ===========================================================================
elif selected_page == "2. Data Understanding":
    render_page_header(
        title="2. Data Understanding",
        subtitle=(
            "Eksplorasi struktur awal dataset, dimensi baris dan kolom, cuplikan tabel data (head), "
            "serta kamus deskripsi ketujuh variabel wajib."
        ),
        badge="Tahap 2 • Pemahaman Data",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Jumlah Baris Data", f"{len(df_raw):,}", "Observasi nasabah bank", "navy")
    with c2:
        render_kpi_card("Jumlah Kolom Wajib", f"{len(df_raw.columns)}", "Sesuai metodologi panduan", "teal")
    with c3:
        render_kpi_card("Fitur Prediktor Awal", "5 Fitur", "Di luar CustomerId & Exited", "blue")
    with c4:
        render_kpi_card("Variabel Target", "Exited (0 / 1)", f"Churn: {kpi_summary['churn_count']:,} baris", "danger")

    if ignored_columns:
        st.info(
            f"ℹ️ **Catatan Metodologi Panduan:** File sumber memiliki kolom tambahan "
            f"(`{', '.join(ignored_columns)}`), namun kolom-kolom tersebut **diabaikan secara otomatis** agar analisis "
            f"mengikuti secara ketat 7 kolom wajib pada panduan Final Project DigitalSkola."
        )

    render_section_title("Cuplikan Data Awal — df.head()")
    num_rows_preview = st.slider(
        "Pilih jumlah baris awal yang ingin ditampilkan:",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
    st.dataframe(df_raw.head(num_rows_preview), use_container_width=True, hide_index=False)

    render_section_title("Kamus Data & Deskripsi Tiap Kolom")
    desc_rows = []
    for col_name in REQUIRED_COLUMNS:
        col_type, col_desc = COLUMN_DESCRIPTIONS.get(col_name, ("-", "-"))
        sample_vals = df_raw[col_name].dropna().unique()[:4]
        sample_str = ", ".join(str(v) for v in sample_vals)
        desc_rows.append(
            {
                "Nama Kolom": col_name,
                "Tipe Data (dtype)": str(df_raw[col_name].dtype),
                "Kategori Variabel": col_type,
                "Contoh Nilai": sample_str,
                "Deskripsi Bisnis": col_desc,
            }
        )
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

    render_section_title("Distribusi Nilai Unik per Kolom")
    uniq_cols = st.columns(2)
    with uniq_cols[0]:
        gender_dist = df_raw["Gender"].value_counts().reset_index()
        gender_dist.columns = ["Gender", "Jumlah Nasabah"]
        gender_dist["Persentase (%)"] = (gender_dist["Jumlah Nasabah"] / len(df_raw) * 100).round(2)
        st.markdown("**Distribusi Variabel `Gender`:**")
        st.dataframe(gender_dist, use_container_width=True, hide_index=True)
    with uniq_cols[1]:
        card_dist = df_raw["HasCrCard"].value_counts().reset_index()
        card_dist.columns = ["HasCrCard", "Jumlah Nasabah"]
        card_dist["Keterangan"] = card_dist["HasCrCard"].map({1: "Punya Kartu Kredit", 0: "Tidak Punya Kartu Kredit"})
        card_dist["Persentase (%)"] = (card_dist["Jumlah Nasabah"] / len(df_raw) * 100).round(2)
        st.markdown("**Distribusi Variabel `HasCrCard`:**")
        st.dataframe(card_dist[["HasCrCard", "Keterangan", "Jumlah Nasabah", "Persentase (%)"]], use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 3: DATA PREPARATION
# ===========================================================================
elif selected_page == "3. Data Preparation":
    render_page_header(
        title="3. Data Preparation",
        subtitle=(
            "Pembersihan atribut yang tidak relevan (drop CustomerId), pemeriksaan tipe data df.info(), "
            "verifikasi missing values df.isnull().sum(), serta statistik deskriptif numerik dan kategorikal."
        ),
        badge="Tahap 3 • Persiapan Data",
    )

    render_section_title("1. Penghapusan Kolom Tidak Relevan — df.drop('CustomerId')")
    col_p1, col_p2 = st.columns([1.2, 1])
    with col_p1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>🧹 Mengapa <code>CustomerId</code> Dihapus?</h4>
                <p>
                    Kolom <code>CustomerId</code> merupakan pengidentifikasi unik setiap nasabah yang bersifat administratif
                    dan tidak memiliki hubungan kausal maupun informasi perilaku terhadap keputusan nasabah menutup rekening (<i>churn</i>).
                    Menyertakan kolom ID unik ke dalam pelatihan model dapat memicu <i>overfitting</i> (model menghafal ID).
                </p>
                <p style="margin-top:0.6rem;">
                    • <b>Kolom Sebelum Drop ({len(df_raw.columns)} kolom):</b> <code>{', '.join(df_raw.columns)}</code><br>
                    • <b>Kolom Sesudah Drop ({len(df_prep.columns)} kolom):</b> <code>{', '.join(df_prep.columns)}</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2:
        st.markdown("**Cuplikan 5 Baris Pertama Setelah Drop `CustomerId`:**")
        st.dataframe(df_prep.head(5), use_container_width=True)

    render_section_title("2. Informasi Tipe Data & Pengecekan Missing Value — df.info() & df.isnull().sum()")
    info_df = get_info_dataframe(df_prep)
    total_nulls = int(df_prep.isnull().sum().sum())

    m1, m2, m3 = st.columns(3)
    with m1:
        render_kpi_card("Total Missing Value", f"{total_nulls}", "Dari seluruh 6 kolom aktif", "teal")
    with m2:
        render_kpi_card(
            "Kolom Numerik",
            f"{len(df_prep.select_dtypes(include=['number']).columns)} Kolom",
            "Age, CreditScore, EstimatedSalary, HasCrCard, Exited",
            "navy",
        )
    with m3:
        render_kpi_card(
            "Kolom Kategorikal",
            f"{len(df_prep.select_dtypes(include=['object']).columns)} Kolom",
            "Gender (Female / Male)",
            "gold",
        )

    st.dataframe(info_df, use_container_width=True, hide_index=True)

    if total_nulls == 0:
        render_insight_box(
            f"Hasil pemeriksaan <code>df.isnull().sum()</code> menunjukkan <b>0 missing value</b> pada seluruh "
            f"{len(df_prep):,} baris dan {len(df_prep.columns)} kolom, sehingga tidak diperlukan imputasi nilai kosong."
        )

    render_section_title("3. Statistik Deskriptif — df.describe()")
    tab_num, tab_cat = st.tabs(["🔢 Statistik Numerik (describe)", "🔤 Statistik Kategorikal (include='object')"])
    with tab_num:
        desc_num = df_prep.describe().T.reset_index().rename(columns={"index": "Atribut Numerik"})
        for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            if c in desc_num.columns:
                desc_num[c] = desc_num[c].round(2)
        st.dataframe(desc_num, use_container_width=True, hide_index=True)
        render_insight_box(
            f"Rentang <b>CreditScore</b> berada pada <b>{int(df_prep['CreditScore'].min())} – {int(df_prep['CreditScore'].max())}</b> "
            f"(rata-rata <b>{df_prep['CreditScore'].mean():.2f}</b>), rentang <b>Age</b> antara "
            f"<b>{int(df_prep['Age'].min())} – {int(df_prep['Age'].max())} tahun</b> (rata-rata <b>{df_prep['Age'].mean():.2f} tahun</b>), "
            f"dan <b>EstimatedSalary</b> berkisar antara <b>${df_prep['EstimatedSalary'].min():,.2f} – ${df_prep['EstimatedSalary'].max():,.2f}</b>."
        )
    with tab_cat:
        desc_cat = df_prep.describe(include=["object"]).T.reset_index().rename(columns={"index": "Atribut Kategorikal"})
        st.dataframe(desc_cat, use_container_width=True, hide_index=True)
        top_gender = str(desc_cat.loc[0, "top"])
        freq_gender = int(desc_cat.loc[0, "freq"])
        render_insight_box(
            f"Atribut <b>Gender</b> memiliki 2 kategori unik dengan nilai terbanyak (<i>modus</i>) adalah "
            f"<b>{top_gender}</b> sebanyak <b>{freq_gender:,} nasabah ({freq_gender / len(df_prep) * 100:.2f}%)</b>."
        )


# ===========================================================================
# HALAMAN 4: EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================================================
elif selected_page == "4. Exploratory Data Analysis":
    render_page_header(
        title="4. Exploratory Data Analysis (EDA) & Feature Engineering",
        subtitle=(
            "Analisis mendalam proporsi churn, distribusi variabel kategorikal & kontinu, One-Hot Encoding, "
            "matriks korelasi sebelum & sesudah rekayasa fitur Credit Score Given Age."
        ),
        badge="Tahap 4 • Eksplorasi & Rekayasa Fitur",
    )

    # 1. Pie chart proporsi Churn vs Retained
    render_section_title("1. Proporsi Nasabah Churn (Exited = 1) vs Bertahan (Exited = 0)")
    col_pie, col_pie_stat = st.columns([1.3, 1])
    with col_pie:
        st.plotly_chart(plot_churn_pie(df_prep), use_container_width=True)
    with col_pie_stat:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>📌 Ketidakseimbangan Kelas (Imbalanced Dataset)</h4>
                <p>
                    • <b>Nasabah Bertahan (Exited = 0):</b> <b>{kpi_summary['retained_count']:,} nasabah ({kpi_summary['retained_rate']:.2f}%)</b><br>
                    • <b>Nasabah Churn (Exited = 1):</b> <b>{kpi_summary['churn_count']:,} nasabah ({kpi_summary['churn_rate']:.2f}%)</b>
                </p>
                <p style="margin-top:0.65rem;">
                    Proporsi kelas target sangat timpang (sekitar <b>4 : 1</b>). Apabila model dilatih langsung tanpa penanganan
                    ketidakseimbangan data, model cenderung bias ke kelas mayoritas (<i>Retained</i>) dan menghasilkan
                    nilai <i>Recall</i> yang rendah pada kelas <i>Churn</i>. Oleh karena itu, teknik <b>SMOTE</b> diterapkan pada tahap pemodelan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_insight_box(
        f"Dari total <b>{kpi_summary['total_customers']:,}</b> nasabah, sebesar <b>{kpi_summary['churn_rate']:.2f}% ({kpi_summary['churn_count']:,} nasabah)</b> "
        f"menutup rekeningnya, sedangkan <b>{kpi_summary['retained_rate']:.2f}% ({kpi_summary['retained_count']:,} nasabah)</b> tetap bertahan."
    )

    # 2. Categorical variables: HasCrCard & Gender
    render_section_title("2. Analisis Variabel Kategorikal (HasCrCard & Gender) terhadap Status Churn")
    cat_c1, cat_c2 = st.columns(2)

    # Hitung churn rate aktual per kategori HasCrCard
    cr_group = df_prep.groupby("HasCrCard")["Exited"].agg(["count", "sum", "mean"]).reset_index()
    cr_group["Churn Rate (%)"] = (cr_group["mean"] * 100).round(2)
    rate_no_card = float(cr_group.loc[cr_group["HasCrCard"] == 0, "Churn Rate (%)"].values[0])
    rate_has_card = float(cr_group.loc[cr_group["HasCrCard"] == 1, "Churn Rate (%)"].values[0])

    # Hitung churn rate aktual per kategori Gender
    gen_group = df_prep.groupby("Gender")["Exited"].agg(["count", "sum", "mean"]).reset_index()
    gen_group["Churn Rate (%)"] = (gen_group["mean"] * 100).round(2)
    rate_female = float(gen_group.loc[gen_group["Gender"] == "Female", "Churn Rate (%)"].values[0])
    rate_male = float(gen_group.loc[gen_group["Gender"] == "Male", "Churn Rate (%)"].values[0])

    with cat_c1:
        st.plotly_chart(plot_categorical_countplot(df_prep, "HasCrCard", "Kepemilikan Kartu Kredit (HasCrCard)"), use_container_width=True)
        cr_display = cr_group.rename(
            columns={"HasCrCard": "HasCrCard", "count": "Total Nasabah", "sum": "Jumlah Churn"}
        )[["HasCrCard", "Total Nasabah", "Jumlah Churn", "Churn Rate (%)"]]
        cr_display["Kategori"] = cr_display["HasCrCard"].map({0: "Tidak Punya (0)", 1: "Punya Kartu Kredit (1)"})
        st.dataframe(cr_display[["Kategori", "Total Nasabah", "Jumlah Churn", "Churn Rate (%)"]], use_container_width=True, hide_index=True)
        render_insight_box(
            f"Nasabah yang memiliki kartu kredit (<b>70.55%</b> dari populasi) memiliki <i>churn rate</i> sebesar "
            f"<b>{rate_has_card:.2f}%</b>, sedangkan nasabah tanpa kartu kredit memiliki <i>churn rate</i> <b>{rate_no_card:.2f}%</b>. "
            f"Selisih yang sangat kecil ({abs(rate_has_card - rate_no_card):.2f}%) menunjukkan kepemilikan kartu kredit bukan pembeda kuat terhadap churn."
        )

    with cat_c2:
        st.plotly_chart(plot_categorical_countplot(df_prep, "Gender", "Jenis Kelamin (Gender)"), use_container_width=True)
        gen_display = gen_group.rename(
            columns={"Gender": "Gender", "count": "Total Nasabah", "sum": "Jumlah Churn"}
        )[["Gender", "Total Nasabah", "Jumlah Churn", "Churn Rate (%)"]]
        st.dataframe(gen_display, use_container_width=True, hide_index=True)
        render_insight_box(
            f"Nasabah perempuan (<b>Female</b>) memiliki tingkat <i>churn</i> sebesar <b>{rate_female:.2f}%</b>, "
            f"jauh lebih tinggi dibandingkan nasabah laki-laki (<b>Male</b>) yang hanya sebesar <b>{rate_male:.2f}%</b> "
            f"(selisih <b>{rate_female - rate_male:.2f} poin persentase</b>)."
        )

    # 3. Continuous attributes: Boxplot CreditScore, Age, EstimatedSalary per Exited
    render_section_title("3. Analisis Variabel Numerik Kontinu (Boxplot per Status Exited)")
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

    with box_c1:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "CreditScore", "Credit Score"), use_container_width=True)
        render_insight_box(
            f"Median <b>CreditScore</b> nasabah Churn adalah <b>{med_cs_1:.0f}</b> (rata-rata {mean_cs_1:.1f}), "
            f"sedikit lebih rendah dibanding nasabah Retained dengan median <b>{med_cs_0:.0f}</b> (rata-rata {mean_cs_0:.1f}). "
            f"Terdapat sejumlah <i>outlier</i> skor kredit sangat rendah (&lt;400) pada kelompok Churn."
        )
    with box_c2:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "Age", "Usia Nasabah (Age)"), use_container_width=True)
        render_insight_box(
            f"Perbedaan mencolok terlihat pada <b>Age</b>: nasabah Churn memiliki median usia <b>{med_age_1:.0f} tahun</b> "
            f"(rata-rata <b>{mean_age_1:.1f} tahun</b>), jauh lebih tua dibanding nasabah Retained dengan median "
            f"<b>{med_age_0:.0f} tahun</b> (rata-rata <b>{mean_age_0:.1f} tahun</b>). Usia adalah pembeda paling signifikan."
        )
    with box_c3:
        st.plotly_chart(plot_continuous_boxplot(df_prep, "EstimatedSalary", "Perkiraan Gaji (EstimatedSalary)"), use_container_width=True)
        render_insight_box(
            f"Distribusi <b>EstimatedSalary</b> hampir identik antara kedua kelompok: median gaji nasabah Churn sebesar "
            f"<b>${med_sal_1:,.2f}</b> dan nasabah Retained sebesar <b>${med_sal_0:,.2f}</b>, menandakan gaji saja tidak memicu churn."
        )

    # 4. Encoding & Correlation Matrix SEBELUM Feature Engineering
    render_section_title("4. One-Hot Encoding & Matriks Korelasi SEBELUM Feature Engineering")
    enc_c1, enc_c2 = st.columns([1, 1.35])
    with enc_c1:
        st.markdown("**Hasil `pd.get_dummies(df, drop_first=True)` (Muncul Kolom `Gender_Male`):**")
        st.dataframe(df_encoded.head(8), use_container_width=True)
        st.caption(
            "Keterangan: Variabel kategorikal `Gender` diubah menjadi biner `Gender_Male` (1 = Laki-laki, 0 = Perempuan) "
            "agar dapat dihitung matriks korelasinya dan dimasukkan ke algoritma Machine Learning."
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
    render_insight_box(
        f"Sebelum rekayasa fitur, variabel yang memiliki korelasi paling kuat dengan <b>Exited</b> adalah "
        f"<b>Age</b> (korelasi positif <b>r = {corr_age_exited:.4f}</b>, semakin tua usia semakin rentan churn) dan "
        f"<b>Gender_Male</b> (korelasi negatif <b>r = {corr_gender_exited:.4f}</b>, laki-laki lebih jarang churn dibanding perempuan). "
        f"Sementara itu, <b>CreditScore</b> memiliki korelasi langsung yang lemah terhadap Exited (<b>r = {corr_cs_exited:.4f}</b>)."
    )

    # 5. Feature Engineering & Correlation Matrix SETELAH Feature Engineering
    render_section_title("5. Feature Engineering (`Credit Score Given Age`) & Matriks Korelasi SETELAH Feature Engineering")
    fe_c1, fe_c2 = st.columns([1, 1.35])
    with fe_c1:
        st.markdown("**Rumus Fitur Baru:** `Credit Score Given Age = CreditScore / Age`")
        st.dataframe(df_fe.head(8), use_container_width=True)
        mean_csga_0 = float(df_fe[df_fe["Exited"] == 0]["Credit Score Given Age"].mean())
        mean_csga_1 = float(df_fe[df_fe["Exited"] == 1]["Credit Score Given Age"].mean())
        st.markdown(
            f"""
            <div class="info-card" style="margin-top:0.5rem;">
                <h4>🔬 Rasionalitas Fitur <code>Credit Score Given Age</code></h4>
                <p>
                    Rasio skor kredit terhadap usia mengukur seberapa tinggi kelayakan kredit yang berhasil dicapai nasabah
                    relatif terhadap usianya.<br>
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
            plot_correlation_heatmap(corr_after, "Correlation Matrix SETELAH Feature Engineering"),
            use_container_width=True,
        )

    corr_csga_exited = float(corr_after.loc["Credit Score Given Age", "Exited"])
    corr_csga_age = float(corr_after.loc["Credit Score Given Age", "Age"])
    render_insight_box(
        f"Setelah penambahan fitur <b>Credit Score Given Age</b>, fitur baru ini menunjukkan korelasi negatif yang kuat "
        f"terhadap <b>Exited</b> sebesar <b>r = {corr_csga_exited:.4f}</b> (jauh lebih kuat dibandingkan korelasi <i>CreditScore</i> murni "
        f"yang hanya <b>r = {corr_cs_exited:.4f}</b>), serta berkorelasi negatif dengan <b>Age</b> sebesar <b>r = {corr_csga_age:.4f}</b>. "
        f"Artinya, nasabah berusia lanjut dengan skor kredit stagnan/rendah (rasio <i>Credit Score Given Age</i> rendah) memiliki risiko churn yang tinggi."
    )


# ===========================================================================
# HALAMAN 5: MODELLING & EVALUATION
# ===========================================================================
elif selected_page == "5. Modelling & Evaluation":
    render_page_header(
        title="5. Modelling & Evaluation",
        subtitle=(
            "Pembagian data Train/Test (75:25, Stratified), pelatihan 4 algoritma klasifikasi pada Skenario A (Sebelum SMOTE) "
            "dan Skenario B (Sesudah SMOTE pada data train), evaluasi kurva ROC, serta analisis Feature Importance."
        ),
        badge="Tahap 5 • Pemodelan & Evaluasi",
    )

    split_info = metrics_payload["split_info"]

    # 1. Informasi Split Data & Distribusi Kelas SMOTE
    render_section_title("1. Pembagian Data (Train-Test Split) & Penanganan Imbalance dengan SMOTE")
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
        total_sm1 = split_info["train_class_dist_smote_1_1"]["0"] + split_info["train_class_dist_smote_1_1"]["1"]
        render_kpi_card(
            "Train Sesudah SMOTE (1:1)",
            f"{total_sm1:,} Baris",
            f"Retained: {split_info['train_class_dist_smote_1_1']['0']:,} | Churn: {split_info['train_class_dist_smote_1_1']['1']:,}",
            "gold",
        )
    with s4:
        best_info = metrics_payload["best_overall"]
        render_kpi_card(
            "Model Terbaik (ROC AUC)",
            best_info["model_name"],
            f"{best_info['scenario']} (AUC: {best_info['roc_auc']:.4f})",
            "danger",
        )

    st.markdown(
        f"""
        <div class="info-card">
            <h4>🛡️ Pencegahan Data Leakage pada Penerapan SMOTE</h4>
            <p>
                Pembagian data dilakukan menggunakan <code>train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)</code>
                pada <b>{split_info['num_features']} fitur prediktor</b> (<code>{', '.join(split_info['feature_columns'])}</code>).
                Sesuai kaidah ilmiah <i>Machine Learning</i>, teknik <b>SMOTE (Synthetic Minority Over-sampling Technique)</b>
                <b>hanya diterapkan pada data latih (<code>X_train, y_train</code>) setelah proses split</b>, sedangkan data uji
                (<code>X_test, y_test</code> sebanyak {split_info['test_rows']:,} baris) dibiarkan murni agar evaluasi mencerminkan
                distribusi dunia nyata secara objektif.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Detail Classification Report & Confusion Matrix per Model & Skenario
    render_section_title("2. Evaluasi Detail Tiap Model (Classification Report, Confusion Matrix & ROC AUC pada Data Test)")
    tab_scen_a, tab_scen_b = st.tabs(
        ["🅰️ Skenario A: SEBELUM SMOTE (Data Train Asli)", "🅱️ Skenario B: SESUDAH SMOTE 1:1 (Pada Data Train)"]
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
        _render_scenario_details(metrics_payload["after_smote_1_1"], "Sesudah SMOTE 1:1")

    # 3. Kurva ROC Gabungan Keempat Model (Sebelum vs Sesudah SMOTE)
    render_section_title("3. Kurva ROC Gabungan Keempat Model (Sebelum SMOTE vs Sesudah SMOTE)")
    roc_c1, roc_c2 = st.columns(2)
    with roc_c1:
        st.plotly_chart(
            plot_combined_roc_curve(metrics_payload["before_smote"], "Sebelum SMOTE"),
            use_container_width=True,
        )
    with roc_c2:
        st.plotly_chart(
            plot_combined_roc_curve(metrics_payload["after_smote_1_1"], "Sesudah SMOTE (1:1)"),
            use_container_width=True,
        )

    gb_auc_before = metrics_payload["before_smote"]["Gradient Boosting"]["roc_auc"]
    gb_auc_after = metrics_payload["after_smote_1_1"]["Gradient Boosting"]["roc_auc"]
    rf_auc_before = metrics_payload["before_smote"]["Random Forest"]["roc_auc"]
    rf_auc_after = metrics_payload["after_smote_1_1"]["Random Forest"]["roc_auc"]
    render_insight_box(
        f"Berdasarkan kurva ROC pada data uji (<code>X_test</code>), <b>Gradient Boosting</b> menempati posisi teratas baik "
        f"sebelum SMOTE (<b>AUC = {gb_auc_before:.4f}</b>) maupun sesudah SMOTE (<b>AUC = {gb_auc_after:.4f}</b>), disusul oleh "
        f"<b>Random Forest</b> (<b>AUC = {rf_auc_before:.4f}</b> sebelum SMOTE dan <b>{rf_auc_after:.4f}</b> sesudah SMOTE). "
        f"Keempat model berada di atas garis tebakan acak (AUC = 0.50)."
    )

    # 4. Tabel Perbandingan Ringkas & Analisis Trade-off Recall vs Accuracy
    render_section_title("4. Tabel Perbandingan Ringkas Semua Model & Analisis Trade-off Recall vs Accuracy")
    comp_df = build_comparison_dataframe(metrics_payload, include_60_40=False)
    max_auc_val = comp_df["ROC AUC"].max()
    comp_df["Status Model"] = comp_df["ROC AUC"].apply(
        lambda v: "⭐ Terbaik (ROC AUC Tertinggi)" if v == max_auc_val else "Kandidat"
    )

    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC AUC"],
            color="#D1FAE5",
        ),
        use_container_width=True,
        hide_index=True,
    )

    gb_rec_before = metrics_payload["before_smote"]["Gradient Boosting"]["recall_churn"]
    gb_rec_after = metrics_payload["after_smote_1_1"]["Gradient Boosting"]["recall_churn"]
    gb_acc_before = metrics_payload["before_smote"]["Gradient Boosting"]["accuracy"]
    gb_acc_after = metrics_payload["after_smote_1_1"]["accuracy"] if "accuracy" in metrics_payload["after_smote_1_1"] else metrics_payload["after_smote_1_1"]["Gradient Boosting"]["accuracy"]
    rf_rec_before = metrics_payload["before_smote"]["Random Forest"]["recall_churn"]
    rf_rec_after = metrics_payload["after_smote_1_1"]["Random Forest"]["recall_churn"]

    st.markdown(
        f"""
        <div class="info-card" style="border-left: 5px solid {COLORS['navy']}; margin-top: 0.8rem;">
            <h4>⚖️ Analisis Trade-Off: Recall (Churn) vs Accuracy Sebelum & Sesudah SMOTE</h4>
            <p>
                1. <b>Sebelum SMOTE (Data Tidak Seimbang):</b> Model mencapai <i>Accuracy</i> tinggi (misalnya <b>Gradient Boosting {gb_acc_before*100:.2f}%</b>),
                tetapi sangat konservatif dalam menebak kelas <i>Churn</i> sehingga <b>Recall (Churn) rendah ({gb_rec_before*100:.2f}% pada Gradient Boosting dan {rf_rec_before*100:.2f}% pada Random Forest)</b>.
                Artinya, lebih dari 70% nasabah yang sebenarnya akan menutup rekening gagal terdeteksi (<i>False Negative</i> tinggi).<br><br>
                2. <b>Sesudah SMOTE (1:1 pada Data Train):</b> Kemampuan mendeteksi nasabah <i>churn</i> melonjak drastis!
                <b>Recall (Churn) pada Gradient Boosting meningkat dari {gb_rec_before*100:.2f}% menjadi {gb_rec_after*100:.2f}%</b>, dan pada
                <b>Random Forest meningkat dari {rf_rec_before*100:.2f}% menjadi {rf_rec_after*100:.2f}%</b>. Konsekuensi alami dari peningkatan sensitivitas ini adalah
                sedikit penurunan pada <i>Accuracy</i> keseluruhan (menjadi {gb_acc_after*100:.2f}% pada Gradient Boosting) karena munculnya sebagian <i>False Positive</i>.<br><br>
                3. <b>Perspektif Bisnis Bank:</b> Dalam kasus retensi nasabah, <b>biaya kehilangan nasabah (False Negative) jauh lebih mahal</b>
                dibandingkan biaya mengirimkan promosi retensi kepada nasabah yang ternyata bertahan (False Positive). Oleh sebab itu,
                model dengan <b>ROC AUC tinggi sekaligus Recall yang seimbang pasca-SMOTE</b> jauh lebih bernilai secara operasional.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Feature Importance dari Random Forest
    render_section_title("5. Tingkat Kepentingan Fitur (Feature Importance — Random Forest)")
    fi_scen_choice = st.radio(
        "Pilih skenario model Random Forest untuk melihat Feature Importance:",
        options=["Sebelum SMOTE", "Sesudah SMOTE (1:1)"],
        horizontal=True,
    )
    fi_data = (
        metrics_payload["before_smote"]["Random Forest"]["feature_importances"]
        if fi_scen_choice == "Sebelum SMOTE"
        else metrics_payload["after_smote_1_1"]["Random Forest"]["feature_importances"]
    )

    fi_col1, fi_col2 = st.columns([1.35, 1])
    with fi_col1:
        st.plotly_chart(
            plot_feature_importance(fi_data, f"Feature Importance Random Forest ({fi_scen_choice})"),
            use_container_width=True,
        )
    with fi_col2:
        sorted_fi = sorted(fi_data.items(), key=lambda x: x[1], reverse=True)
        fi_table = pd.DataFrame(
            [
                {
                    "Peringkat": i + 1,
                    "Nama Fitur": k,
                    "Bobot Importance": round(v, 4),
                    "Kontribusi (%)": f"{v*100:.2f}%",
                }
                for i, (k, v) in enumerate(sorted_fi)
            ]
        )
        st.markdown("**Tabel Peringkat Kontribusi Fitur:**")
        st.dataframe(fi_table, use_container_width=True, hide_index=True)

    top1_name, top1_val = sorted_fi[0]
    top2_name, top2_val = sorted_fi[1]
    top3_name, top3_val = sorted_fi[2]
    render_insight_box(
        f"Pada model Random Forest ({fi_scen_choice}), tiga fitur yang paling berpengaruh dalam menentukan keputusan nasabah "
        f"churn adalah <b>1) {top1_name} ({top1_val*100:.2f}%)</b>, <b>2) {top2_name} ({top2_val*100:.2f}%)</b>, dan "
        f"<b>3) {top3_name} ({top3_val*100:.2f}%)</b>. Sementara itu, atribut biner <b>Gender_Male</b> dan <b>HasCrCard</b> "
        f"memiliki kontribusi paling kecil (masing-masing di bawah 2%). Ini membuktikan bahwa fitur hasil rekayasa "
        f"<code>Credit Score Given Age</code> bersama variabel kontinu mampu mendominasi daya pisah pohon keputusan."
    )


# ===========================================================================
# HALAMAN 6: KESIMPULAN, REKOMENDASI, DAN SARAN
# ===========================================================================
elif selected_page == "6. Kesimpulan & Saran":
    render_page_header(
        title="6. Kesimpulan, Rekomendasi & Eksperimen Saran SMOTE",
        subtitle=(
            "Menjawab ketiga objective bisnis berdasarkan angka aktual hasil analisis, rekomendasi strategi retensi bank, "
            "serta simulasi interaktif saran panduan: perbandingan rasio SMOTE 1:1 (50:50) vs 60:40."
        ),
        badge="Tahap 6 • Kesimpulan & Strategi",
    )

    # Ambil angka aktual untuk narasi kesimpulan
    corr_after = df_fe.corr(numeric_only=True)
    corr_age = float(corr_after.loc["Age", "Exited"])
    corr_csga = float(corr_after.loc["Credit Score Given Age", "Exited"])
    corr_gender = float(corr_after.loc["Gender_Male", "Exited"])

    fi_before = sorted(
        metrics_payload["before_smote"]["Random Forest"]["feature_importances"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    fi_after = sorted(
        metrics_payload["after_smote_1_1"]["Random Forest"]["feature_importances"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    best_overall = metrics_payload["best_overall"]

    # Segmen risiko berdasarkan kelompok umur aktual
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

    render_section_title("1. Kesimpulan Utama (Menjawab Tiga Objective Bisnis)")
    c_obj1, c_obj2, c_obj3 = st.columns(3)

    with_fi_1 = f"{fi_before[0][0]} ({fi_before[0][1]*100:.2f}%)"
    with_fi_2 = f"{fi_before[1][0]} ({fi_before[1][1]*100:.2f}%)"
    with_fi_3 = f"{fi_before[2][0]} ({fi_before[2][1]*100:.2f}%)"

    with c_obj1:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['navy']}; height: 100%;">
                <h4>🔍 Jawaban Objective (a): Faktor Paling Memengaruhi Churn</h4>
                <p>
                    1. <b>Dari Analisis Korelasi & Demografi:</b> Faktor terkuat yang memicu <i>churn</i> adalah <b>Usia (Age)</b>
                    dengan korelasi positif <b>r = {corr_age:.4f}</b> dan fitur rekayasa <b>Credit Score Given Age</b> dengan korelasi
                    negatif <b>r = {corr_csga:.4f}</b>. Segmen usia <b>{highest_risk_seg['Kelompok Usia']}</b> mencatat tingkat
                    <i>churn</i> tertinggi yaitu <b>{highest_risk_seg['Churn Rate (%)']:.2f}%</b>. Selain itu, nasabah perempuan memiliki
                    proporsi <i>churn</i> lebih tinggi (<b>25.07%</b>) dibanding laki-laki (<b>16.46%</b>).<br><br>
                    2. <b>Dari Feature Importance Random Forest:</b> Tiga fitur kontributor terbesar pada pemisahan pohon keputusan adalah
                    <b>{with_fi_1}</b>, <b>{with_fi_2}</b>, dan <b>{with_fi_3}</b> (atau <b>Age {fi_after[0][1]*100:.2f}%</b> pasca-SMOTE).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_obj2:
        gb_b = metrics_payload["before_smote"]["Gradient Boosting"]
        gb_a = metrics_payload["after_smote_1_1"]["Gradient Boosting"]
        rf_b = metrics_payload["before_smote"]["Random Forest"]
        rf_a = metrics_payload["after_smote_1_1"]["Random Forest"]
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['teal']}; height: 100%;">
                <h4>🏆 Jawaban Objective (b): Model Prediksi Paling Akurat</h4>
                <p>
                    1. <b>Berdasarkan ROC AUC & Accuracy:</b> Algoritma <b>{best_overall['model_name']} ({best_overall['scenario']})</b>
                    merupakan model terbaik secara keseluruhan dengan <b>ROC AUC = {best_overall['roc_auc']:.4f}</b> dan
                    <b>Accuracy = {best_overall['accuracy']*100:.2f}%</b>.<br><br>
                    2. <b>Berdasarkan Sensitivitas Deteksi Churn (Recall):</b> Penerapan <b>SMOTE pada data train</b> meningkatkan
                    <i>Recall</i> kelas churn secara signifikan: pada <b>Gradient Boosting</b> naik dari <b>{gb_b['recall_churn']*100:.2f}%</b>
                    menjadi <b>{gb_a['recall_churn']*100:.2f}%</b> (AUC = {gb_a['roc_auc']:.4f}), dan pada <b>Random Forest</b> naik dari
                    <b>{rf_b['recall_churn']*100:.2f}%</b> menjadi <b>{rf_a['recall_churn']*100:.2f}%</b> (AUC = {rf_a['roc_auc']:.4f}).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_obj3:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['gold']}; height: 100%;">
                <h4>🚀 Jawaban Objective (c): Implementasi untuk Pertumbuhan Bisnis</h4>
                <p>
                    1. <b>Early Warning System (Sistem Peringatan Dini):</b> Mengintegrasikan model prediktif ke dalam sistem CRM bank
                    untuk memberi skor risiko <i>churn</i> bulanan pada setiap nasabah aktif.<br><br>
                    2. <b>Intervensi Terfokus pada Segmen Berisiko Tinggi:</b> Memfokuskan anggaran kampanye retensi Divisi Marketing pada
                    nasabah berusia <b>41–60 tahun</b>, nasabah perempuan, serta nasabah dengan rasio <code>Credit Score Given Age</code>
                    rendah melalui produk pensiun/investasi yang lebih kompetitif dan layanan <i>Relationship Manager</i> khusus.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_title("2. Profil Risiko Churn Berdasarkan Kelompok Usia (Data Aktual)")
    st.dataframe(
        seg_summary[["Kelompok Usia", "Total_Nasabah", "Jumlah_Churn", "Churn Rate (%)"]].rename(
            columns={"Total_Nasabah": "Total Nasabah", "Jumlah_Churn": "Jumlah Churn"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 3. Saran Panduan: Eksperimen Perbandingan Rasio SMOTE (1:1 vs sekitar 60:40)
    render_section_title("3. Saran Pengembangan Metodologi: Eksperimen Rasio SMOTE 1:1 (50:50) vs 60:40")
    st.markdown(
        f"""
        <div class="info-card">
            <h4>🧪 Mengapa Mencoba Rasio SMOTE 60:40?</h4>
            <p>
                Pada panduan Final Project DigitalSkola, disarankan untuk mencoba pengaturan rasio <i>oversampling</i> <b>SMOTE sekitar 60:40</b>
                (<code>sampling_strategy = 0.6667</code>, di mana kelas minoritas disintesis hingga mencapai 40% dari total data latih dan
                kelas mayoritas tetap dominan sebesar 60%). Tujuannya adalah menjaga keseimbangan alami agar model tidak terlalu agresif
                menghasilkan <i>False Positive</i>, sehingga <b>Accuracy</b> dan <b>Precision</b> tetap terjaga tinggi sembari mempertahankan
                peningkatan <b>Recall</b> dan <b>ROC AUC</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    smote_view_mode = st.selectbox(
        "Pilih Mode Perbandingan Rasio SMOTE untuk Ditampilkan:",
        options=[
            "Perbandingan Lengkap (Sebelum SMOTE vs SMOTE 1:1 vs SMOTE 60:40)",
            "Fokus SMOTE 60:40 (Mayoritas 60% : Minoritas 40%)",
            "Fokus SMOTE 1:1 (Seimbang 50% : 50%)",
        ],
        index=0,
    )

    dist_60_40 = metrics_payload["split_info"]["train_class_dist_smote_60_40"]
    tot_60_40 = dist_60_40["0"] + dist_60_40["1"]
    dist_1_1 = metrics_payload["split_info"]["train_class_dist_smote_1_1"]
    tot_1_1 = dist_1_1["0"] + dist_1_1["1"]

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        render_kpi_card(
            "Komposisi Data Train — SMOTE 1:1 (50:50)",
            f"{tot_1_1:,} Baris",
            f"Retained (0): {dist_1_1['0']:,} (50.0%) | Churn (1): {dist_1_1['1']:,} (50.0%)",
            "teal",
        )
    with dcol2:
        pct_0 = dist_60_40["0"] / tot_60_40 * 100
        pct_1 = dist_60_40["1"] / tot_60_40 * 100
        render_kpi_card(
            "Komposisi Data Train — SMOTE 60:40",
            f"{tot_60_40:,} Baris",
            f"Retained (0): {dist_60_40['0']:,} ({pct_0:.1f}%) | Churn (1): {dist_60_40['1']:,} ({pct_1:.1f}%)",
            "gold",
        )

    df_all_scen = build_comparison_dataframe(metrics_payload, include_60_40=True)
    if smote_view_mode == "Fokus SMOTE 60:40 (Mayoritas 60% : Minoritas 40%)":
        df_show = df_all_scen[df_all_scen["Skenario"] == "Sesudah SMOTE (60:40)"]
    elif smote_view_mode == "Fokus SMOTE 1:1 (Seimbang 50% : 50%)":
        df_show = df_all_scen[df_all_scen["Skenario"] == "Sesudah SMOTE (1:1)"]
    else:
        df_show = df_all_scen

    st.dataframe(
        df_show.style.highlight_max(
            subset=["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC AUC"],
            color="#FEF3C7",
        ),
        use_container_width=True,
        hide_index=True,
    )

    roc_sm1, roc_sm60 = st.columns(2)
    with roc_sm1:
        st.plotly_chart(
            plot_combined_roc_curve(metrics_payload["after_smote_1_1"], "SMOTE Rasio 1:1 (50:50)"),
            use_container_width=True,
        )
    with roc_sm60:
        st.plotly_chart(
            plot_combined_roc_curve(metrics_payload["after_smote_60_40"], "SMOTE Rasio 60:40"),
            use_container_width=True,
        )

    gb_60 = metrics_payload["after_smote_60_40"]["Gradient Boosting"]
    gb_11 = metrics_payload["after_smote_1_1"]["Gradient Boosting"]
    rf_60 = metrics_payload["after_smote_60_40"]["Random Forest"]
    rf_11 = metrics_payload["after_smote_1_1"]["Random Forest"]

    render_insight_box(
        f"Hasil eksperimen membuktikan saran panduan sangat tepat! Dengan mengubah rasio SMOTE dari <b>1:1</b> menjadi <b>60:40</b>: "
        f"pada <b>Gradient Boosting</b>, <i>Accuracy</i> meningkat dari <b>{gb_11['accuracy']*100:.2f}%</b> menjadi <b>{gb_60['accuracy']*100:.2f}%</b>, "
        f"<i>F1-Score (Churn)</i> naik dari <b>{gb_11['f1_churn']:.4f}</b> menjadi <b>{gb_60['f1_churn']:.4f}</b>, dan <i>ROC AUC</i> naik menjadi "
        f"<b>{gb_60['roc_auc']:.4f}</b>. Begitu pula pada <b>Random Forest</b>, <i>Accuracy</i> meningkat dari <b>{rf_11['accuracy']*100:.2f}%</b> "
        f"menjadi <b>{rf_60['accuracy']*100:.2f}%</b> dan <i>ROC AUC</i> naik dari <b>{rf_11['roc_auc']:.4f}</b> menjadi <b>{rf_60['roc_auc']:.4f}</b>."
    )


# ===========================================================================
# HALAMAN 7: PREDIKSI NASABAH (HALAMAN TAMBAHAN INTERAKTIF)
# ===========================================================================
elif selected_page == "7. Prediksi Nasabah":
    render_page_header(
        title="7. Simulasi Prediksi Risiko Churn Nasabah",
        subtitle=(
            "Masukkan profil demografis dan finansial nasabah untuk menghitung otomatis fitur Credit Score Given Age "
            "dan memprediksi probabilitas churn menggunakan model Random Forest terbaik."
        ),
        badge="Tahap 7 • Kalkulator Prediksi Real-Time",
    )

    col_form, col_result = st.columns([1.05, 1.25])

    with col_form:
        render_section_title("Formulir Profil Nasabah")
        with st.form("churn_prediction_form"):
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                gender_in = st.selectbox(
                    "Jenis Kelamin (Gender)",
                    options=["Female", "Male"],
                    format_func=lambda g: "Perempuan (Female)" if g == "Female" else "Laki-laki (Male)",
                )
                age_in = st.number_input(
                    "Usia Nasabah (Age - Tahun)",
                    min_value=18,
                    max_value=95,
                    value=45,
                    step=1,
                )
                has_card_in = st.selectbox(
                    "Kepemilikan Kartu Kredit (HasCrCard)",
                    options=[1, 0],
                    format_func=lambda v: "1 — Memiliki Kartu Kredit" if v == 1 else "0 — Tidak Memiliki Kartu Kredit",
                )
            with f_c2:
                credit_score_in = st.number_input(
                    "Skor Kredit (CreditScore: 300 – 850)",
                    min_value=300,
                    max_value=850,
                    value=580,
                    step=5,
                )
                salary_in = st.number_input(
                    "Perkiraan Gaji Tahunan (EstimatedSalary - $)",
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
                    help="Default menggunakan model Random Forest terbaik dengan keseimbangan optimal.",
                )

            live_csga = float(credit_score_in) / float(age_in)
            st.info(
                f"⚙️ **Fitur Rekayasa Otomatis (`Credit Score Given Age`):** "
                f"`{credit_score_in} / {age_in}` = **{live_csga:.4f}**"
            )

            submitted = st.form_submit_button(
                "🔍 Prediksi Probabilitas Churn Nasabah",
                use_container_width=True,
                type="primary",
            )

    # Jalankan evaluasi prediksi secara langsung (baik saat awal halaman dibuka maupun saat tombol ditekan)
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
        render_section_title("Hasil Prediksi & Rekomendasi Tindakan Retensi")
        st.plotly_chart(plot_churn_probability_gauge(pred_out["prob_churn"]), use_container_width=True)

        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            render_kpi_card(
                "Probabilitas Churn",
                f"{pred_out['prob_churn']*100:.2f}%",
                f"Retensi: {pred_out['prob_retained']*100:.2f}%",
                "danger" if pred_out["risk_category"] == "Tinggi" else "gold" if pred_out["risk_category"] == "Sedang" else "teal",
            )
        with r_c2:
            render_kpi_card(
                "Kategori Risiko",
                f"Risiko {pred_out['risk_category']}",
                "Ambang: <35% Rendah | 35-65% Sedang | >65% Tinggi",
                "danger" if pred_out["risk_category"] == "Tinggi" else "gold" if pred_out["risk_category"] == "Sedang" else "teal",
            )
        with r_c3:
            render_kpi_card(
                "Credit Score Given Age",
                f"{pred_out['credit_score_given_age']:.2f}",
                f"Skor {credit_score_in} / Usia {age_in} thn",
                "navy",
            )

        rec_items_html = "".join(f"<li style='margin-bottom:0.35rem;'>{item}</li>" for item in pred_out["recommendations"])
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 5px solid {pred_out['risk_color']};">
                <h4>📋 Rekomendasi Tindakan Retensi (Kategori Risiko {pred_out['risk_category']})</h4>
                <ul style="margin: 0.4rem 0 0 1.1rem; padding: 0; font-size: 0.92rem; line-height: 1.55;">
                    {rec_items_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_title("Vektor Fitur Input yang Diteruskan ke Model Random Forest")
    st.dataframe(pred_out["input_dataframe"], use_container_width=True, hide_index=True)


# ===========================================================================
# HALAMAN 8: TENTANG (ABOUT)
# ===========================================================================
elif selected_page == "8. Tentang":
    render_page_header(
        title="8. Tentang Aplikasi & Metodologi",
        subtitle=(
            "Ringkasan metodologi CRISP-DM, sumber panduan Final Project DigitalSkola, keterbatasan studi, "
            "dan identitas pengembang."
        ),
        badge="Tahap 8 • Dokumentasi & Kredit",
    )

    col_ab1, col_ab2 = st.columns([1.3, 1])

    with col_ab1:
        render_section_title("Ringkasan Metodologi & Sumber Panduan")
        st.markdown(
            f"""
            <div class="info-card">
                <h4>📚 Sumber Panduan Resmi</h4>
                <p>
                    Aplikasi web ini dikembangkan untuk mereplikasi secara utuh alur analisis pada panduan
                    <b>Final Project: "Bank Customer Churn"</b> yang disusun oleh <b>Datamine Team, DigitalSkola</b>.
                    Seluruh metrik, tabel statistik, matriks korelasi, kurva ROC, dan tingkat kepentingan fitur
                    dihitung secara langsung (<i>real-time/pre-computed</i>) dari dataset <code>data/Churn_Modelling.csv</code>
                    ({len(df_raw):,} baris) tanpa adanya angka yang di-<i>hardcode</i>.
                </p>
                <h4 style="margin-top: 1rem;">🔄 Tahapan Metodologi</h4>
                <p>
                    1. <b>Seleksi Kolom Wajib:</b> Menggunakan 7 atribut sesuai panduan (<code>CustomerId, Gender, Age, CreditScore, EstimatedSalary, HasCrCard, Exited</code>).<br>
                    2. <b>Data Preparation & Encoding:</b> Menghapus <code>CustomerId</code> dan menerapkan <i>One-Hot Encoding</i> (<code>drop_first=True</code>) menghasilkan <code>Gender_Male</code>.<br>
                    3. <b>Feature Engineering:</b> Menciptakan fitur rasio <code>Credit Score Given Age = CreditScore / Age</code>.<br>
                    4. <b>Pemodelan & SMOTE Tanpa Data Leakage:</b> Membagi data dengan <code>train_test_split(test_size=0.25, random_state=42, stratify=y)</code> dan menerapkan <code>SMOTE</code> hanya pada data <i>train</i>.<br>
                    5. <b>Evaluasi Komprehensif:</b> Membandingkan <i>Gradient Boosting, KNN, Random Forest,</i> dan <i>Decision Tree</i> pada data <i>test</i> baik sebelum SMOTE, sesudah SMOTE 1:1, maupun rasio SMOTE 60:40.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Keterbatasan Analisis (Limitations)")
        st.markdown(
            f"""
            <div class="info-card" style="border-left: 4px solid {COLORS['danger']};">
                <h4>⚠️ Keterbatasan Dataset & Ruang Lingkup</h4>
                <p>
                    • <b>Keterbatasan Jumlah Fitur (Hanya 5 Fitur Dasar):</b> Sesuai batasan metodologi panduan, analisis hanya menggunakan
                    5 atribut prediktor awal (<code>Gender, Age, CreditScore, EstimatedSalary, HasCrCard</code>) ditambah 1 fitur rekayasa.
                    Atribut perbankan lain seperti saldo rekening (<i>Balance</i>), jumlah produk (<i>NumOfProducts</i>), keaktifan anggota
                    (<i>IsActiveMember</i>), dan wilayah geografis (<i>Geography</i>) tidak disertakan, yang membatasi plafon maksimal skor akurasi model.<br><br>
                    • <b>Ketidakseimbangan Kelas Target (Class Imbalance):</b> Proporsi nasabah <i>churn</i> hanya sebesar
                    <b>{kpi_summary['churn_rate']:.2f}%</b> berbanding <b>{kpi_summary['retained_rate']:.2f}%</b> nasabah bertahan,
                    sehingga diperlukan kehati-hatian dalam memilih rasio <i>oversampling</i> SMOTE dan ambang batas keputusan (<i>threshold</i>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ab2:
        render_section_title("Profil Pengembang")
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {COLORS['gold']};">
                <h4>👤 Identitas Penyusun Proyek</h4>
                <p>
                    • <b>Nama Lengkap:</b> <code>Afrizal Subagja</code> <span style="color:#6B7280;font-size:0.82rem;">([Nama Anda])</span><br>
                    • <b>NIM / ID Peserta:</b> <code>072926002</code> <span style="color:#6B7280;font-size:0.82rem;">([NIM])</span><br>
                    • <b>Mata Kuliah / Program:</b> Pembelajaran Mesin (Machine Learning) / Data Science<br>
                    • <b>Acuan Studi Kasus:</b> Final Project "Bank Customer Churn" — Datamine Team, DigitalSkola
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_title("Spesifikasi Teknis Aplikasi")
        st.markdown(
            """
            <div class="info-card">
                <h4>🛠️ Stack Teknologi</h4>
                <p>
                    • <b>Framework UI:</b> Streamlit &amp; Streamlit Option Menu<br>
                    • <b>Pemrosesan Data:</b> Pandas &amp; NumPy<br>
                    • <b>Machine Learning:</b> Scikit-Learn &amp; Imbalanced-Learn (SMOTE)<br>
                    • <b>Visualisasi Interaktif:</b> Plotly Express &amp; Graph Objects<br>
                    • <b>Serialisasi Artefak:</b> Joblib (<code>rf_pipeline.pkl</code>) &amp; JSON (<code>metrics.json</code>)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
