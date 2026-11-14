"""Modul styling CSS dan komponen antarmuka (UI) bertema dashboard perbankan modern."""

import streamlit as st

# Palet warna resmi sesuai spesifikasi desain
COLORS = {
    "navy": "#0B2545",
    "blue": "#13315C",
    "teal": "#1B998B",
    "gold": "#F4B942",
    "danger": "#E4572E",
    "bg": "#F5F7FA",
    "card": "#FFFFFF",
    "text_primary": "#1F2937",
    "text_secondary": "#6B7280",
}


def apply_custom_style() -> None:
    """Menyuntikkan CSS kustom ke aplikasi Streamlit untuk tampilan perbankan profesional."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {COLORS["text_primary"]};
        }}

        /* Sembunyikan menu default dan footer Streamlit */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        /* Latar utama aplikasi */
        .stApp {{
            background-color: {COLORS["bg"]};
        }}

        /* Styling Sidebar Navy */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS["navy"]} 0%, {COLORS["blue"]} 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        section[data-testid="stSidebar"] * {{
            color: #F9FAFB !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: #D1D5DB !important;
        }}

        /* Judul Halaman dengan Garis Aksen */
        .page-header-box {{
            background: linear-gradient(135deg, {COLORS["navy"]} 0%, {COLORS["blue"]} 100%);
            color: #FFFFFF;
            padding: 1.6rem 2rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(11, 37, 69, 0.12);
            border-bottom: 4px solid {COLORS["gold"]};
        }}
        .page-header-badge {{
            display: inline-block;
            background-color: rgba(244, 185, 66, 0.2);
            color: {COLORS["gold"]};
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(244, 185, 66, 0.45);
        }}
        .page-header-title {{
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: #FFFFFF !important;
            letter-spacing: -0.02em;
        }}
        .page-header-subtitle {{
            font-size: 0.95rem;
            color: #E5E7EB !important;
            margin: 0;
            font-weight: 400;
            line-height: 1.5;
        }}

        /* Sub-judul Seksi dengan garis aksen kiri */
        .section-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {COLORS["navy"]};
            padding-left: 0.85rem;
            border-left: 4px solid {COLORS["teal"]};
            margin: 1.5rem 0 0.9rem 0;
            line-height: 1.35;
        }}

        /* KPI Cards sudut membulat, shadow halus, dan border-left aksen */
        .kpi-card {{
            background-color: {COLORS["card"]};
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            border: 1px solid #E5E7EB;
            border-left: 5px solid {COLORS["teal"]};
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            height: 100%;
            margin-bottom: 0.75rem;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.09);
        }}
        .kpi-card.navy {{ border-left-color: {COLORS["navy"]}; }}
        .kpi-card.teal {{ border-left-color: {COLORS["teal"]}; }}
        .kpi-card.gold {{ border-left-color: {COLORS["gold"]}; }}
        .kpi-card.danger {{ border-left-color: {COLORS["danger"]}; }}
        .kpi-card.blue {{ border-left-color: {COLORS["blue"]}; }}

        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLORS["text_secondary"]};
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {COLORS["navy"]};
            margin-bottom: 0.25rem;
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.8rem;
            color: {COLORS["text_secondary"]};
            margin: 0;
        }}

        /* Kartu Konten Putih Serbaguna */
        .info-card {{
            background-color: {COLORS["card"]};
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
            border: 1px solid #E5E7EB;
            margin-bottom: 1rem;
        }}
        .info-card h4 {{
            color: {COLORS["navy"]};
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.05rem;
            font-weight: 700;
        }}
        .info-card p {{
            color: {COLORS["text_primary"]};
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 0;
        }}

        /* Kotak Insight di bawah grafik */
        .insight-box {{
            background-color: #FFFFFF;
            border-left: 4px solid {COLORS["gold"]};
            border-top: 1px solid #E5E7EB;
            border-right: 1px solid #E5E7EB;
            border-bottom: 1px solid #E5E7EB;
            padding: 0.95rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin-top: 0.4rem;
            margin-bottom: 1.25rem;
            font-size: 0.9rem;
            color: {COLORS["text_primary"]};
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }}
        .insight-box strong {{
            color: {COLORS["navy"]};
        }}

        /* Styling Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, badge: str = "Bank Customer Churn Analytics") -> None:
    """Merender header halaman dengan gradien navy, badge emas, dan deskripsi."""
    st.markdown(
        f"""
        <div class="page-header-box">
            <div class="page-header-badge">{badge}</div>
            <h1 class="page-header-title">{title}</h1>
            <p class="page-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    """Merender sub-judul bagian dengan garis aksen teal di sebelah kiri."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str = "", accent: str = "teal") -> None:
    """Merender kartu metrik KPI dengan border-left aksen warna sesuai tema."""
    st.markdown(
        f"""
        <div class="kpi-card {accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_box(text: str) -> None:
    """Merender kotak insight analisis data di bawah grafik Plotly."""
    st.markdown(
        f"""
        <div class="insight-box">
            💡 <strong>Insight Data Aktual:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_kpi_row(kpi: dict) -> None:
    """Merender baris 5 kartu KPI utama di halaman awal maupun ringkasan eksekutif."""
    cols = st.columns(5)
    with cols[0]:
        render_kpi_card(
            label="Total Nasabah",
            value=f"{kpi['total_customers']:,}",
            subtext="Populasi sampel nasabah",
            accent="navy",
        )
    with cols[1]:
        render_kpi_card(
            label="Jumlah Churn",
            value=f"{kpi['churn_count']:,}",
            subtext=f"Retained: {kpi['retained_count']:,} nasabah",
            accent="danger",
        )
    with cols[2]:
        render_kpi_card(
            label="Churn Rate",
            value=f"{kpi['churn_rate']:.2f}%",
            subtext=f"Retensi: {kpi['retained_rate']:.2f}%",
            accent="gold",
        )
    with cols[3]:
        render_kpi_card(
            label="Rata-rata Usia",
            value=f"{kpi['avg_age']:.1f} Thn",
            subtext="Usia rata-rata seluruh nasabah",
            accent="teal",
        )
    with cols[4]:
        render_kpi_card(
            label="Rata-rata Credit Score",
            value=f"{kpi['avg_credit_score']:.0f}",
            subtext="Skor kelayakan kredit",
            accent="blue",
        )
