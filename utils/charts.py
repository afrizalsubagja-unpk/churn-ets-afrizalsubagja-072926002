"""Modul pembuatan visualisasi interaktif Plotly dengan tema perbankan konsisten."""

from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.style import COLORS

# Pemetaan warna konsisten untuk status Exited
STATUS_COLOR_MAP = {
    "Retained (0)": COLORS["teal"],
    "Churn (1)": COLORS["danger"],
    0: COLORS["teal"],
    1: COLORS["danger"],
}


def _apply_standard_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Menerapkan tata letak dasar Plotly bertema putih bersih dengan tipografi Inter."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=16, color=COLORS["navy"], weight="bold"),
            x=0.01,
            xanchor="left",
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"], size=12),
        height=height,
        margin=dict(l=45, r=30, t=65, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig


def plot_churn_pie(df: pd.DataFrame) -> go.Figure:
    """Membuat Pie/Donut Chart proporsi nasabah Churn (Exited=1) vs Retained (Exited=0)."""
    counts = df["Exited"].value_counts().rename(index={0: "Retained (0)", 1: "Churn (1)"}).reset_index()
    counts.columns = ["Status Nasabah", "Jumlah Nasabah"]

    fig = px.pie(
        counts,
        names="Status Nasabah",
        values="Jumlah Nasabah",
        color="Status Nasabah",
        color_discrete_map={
            "Retained (0)": COLORS["teal"],
            "Churn (1)": COLORS["danger"],
        },
        hole=0.45,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label+value",
        hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,} nasabah<br>Proporsi: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    return _apply_standard_layout(fig, "Proporsi Nasabah Churn vs Bertahan (Retained)", height=400)


def plot_categorical_countplot(df: pd.DataFrame, col: str, label_id: str) -> go.Figure:
    """
    Membuat grouped bar chart (countplot) untuk atribut kategorikal (HasCrCard atau Gender)
    yang dipisah berdasarkan status Exited, lengkap dengan label jumlah dan persentase.
    """
    temp = df.copy()
    if col == "HasCrCard":
        temp["Kategori"] = temp[col].map({1: "Punya Kartu Kredit (1)", 0: "Tidak Punya (0)"})
    elif col == "Gender":
        temp["Kategori"] = temp[col].map({"Female": "Perempuan (Female)", "Male": "Laki-laki (Male)"})
    else:
        temp["Kategori"] = temp[col].astype(str)

    temp["Status Nasabah"] = temp["Exited"].map({0: "Retained (0)", 1: "Churn (1)"})

    grouped = (
        temp.groupby(["Kategori", "Status Nasabah"], as_index=False)
        .size()
        .rename(columns={"size": "Jumlah Nasabah"})
    )
    total_per_cat = grouped.groupby("Kategori")["Jumlah Nasabah"].transform("sum")
    grouped["Persentase Kategori (%)"] = (grouped["Jumlah Nasabah"] / total_per_cat) * 100.0
    grouped["Label Teks"] = grouped.apply(
        lambda r: f"{int(r['Jumlah Nasabah']):,} ({r['Persentase Kategori (%)']:.1f}%)", axis=1
    )

    fig = px.bar(
        grouped,
        x="Kategori",
        y="Jumlah Nasabah",
        color="Status Nasabah",
        barmode="group",
        text="Label Teks",
        color_discrete_map={
            "Retained (0)": COLORS["teal"],
            "Churn (1)": COLORS["danger"],
        },
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b> - %{fullData.name}<br>Jumlah: %{y:,} nasabah<extra></extra>",
    )
    fig.update_xaxes(title_text=label_id, showgrid=False)
    fig.update_yaxes(title_text="Jumlah Nasabah", gridcolor="#F3F4F6")
    return _apply_standard_layout(fig, f"Distribusi {label_id} Berdasarkan Status Churn", height=420)


def plot_continuous_boxplot(df: pd.DataFrame, col: str, label_id: str) -> go.Figure:
    """
    Membuat Boxplot atribut numerik kontinu (CreditScore, Age, EstimatedSalary)
    per status Exited (Retained vs Churn).
    """
    temp = df.copy()
    temp["Status Nasabah"] = temp["Exited"].map({0: "Retained (0)", 1: "Churn (1)"})

    fig = px.box(
        temp,
        x="Status Nasabah",
        y=col,
        color="Status Nasabah",
        category_orders={"Status Nasabah": ["Retained (0)", "Churn (1)"]},
        color_discrete_map={
            "Retained (0)": COLORS["teal"],
            "Churn (1)": COLORS["danger"],
        },
        points="outliers",
    )
    fig.update_xaxes(title_text="Status Nasabah (Exited)", showgrid=False)
    fig.update_yaxes(title_text=label_id, gridcolor="#F3F4F6")
    return _apply_standard_layout(fig, f"Boxplot {label_id} per Status Churn", height=400)


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    """Membuat visualisasi Correlation Matrix (Heatmap) dengan nilai korelasi 2 desimal."""
    z_vals = np.round(corr_df.values, 3)
    text_vals = [[f"{val:.2f}" for val in row] for row in z_vals]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=list(corr_df.columns),
            y=list(corr_df.index),
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#1F2937"),
            colorscale=[
                [0.0, COLORS["danger"]],
                [0.5, "#F9FAFB"],
                [1.0, COLORS["navy"]],
            ],
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Korelasi (r)"),
            hovertemplate="Fitur X: <b>%{x}</b><br>Fitur Y: <b>%{y}</b><br>Korelasi: <b>%{z:.4f}</b><extra></extra>",
        )
    )
    fig.update_xaxes(side="bottom", tickangle=-25)
    fig.update_yaxes(autorange="reversed")
    return _apply_standard_layout(fig, title, height=460)


def plot_confusion_matrix(cm: list, model_name: str, scenario_label: str) -> go.Figure:
    """Membuat heatmap Confusion Matrix 2x2 pada data uji (X_test)."""
    cm_arr = np.array(cm)
    total = cm_arr.sum()
    labels_x = ["Prediksi Retained (0)", "Prediksi Churn (1)"]
    labels_y = ["Aktual Retained (0)", "Aktual Churn (1)"]

    annotations_text = [
        [
            f"<b>TN: {cm_arr[0, 0]:,}</b><br>({cm_arr[0, 0]/total*100:.1f}%)",
            f"<b>FP: {cm_arr[0, 1]:,}</b><br>({cm_arr[0, 1]/total*100:.1f}%)",
        ],
        [
            f"<b>FN: {cm_arr[1, 0]:,}</b><br>({cm_arr[1, 0]/total*100:.1f}%)",
            f"<b>TP: {cm_arr[1, 1]:,}</b><br>({cm_arr[1, 1]/total*100:.1f}%)",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=labels_x,
            y=labels_y,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale=[
                [0.0, "#F0FDFA"],
                [0.5, "#5EEAD4"],
                [1.0, COLORS["navy"]],
            ],
            showscale=False,
            hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: <b>%{z:,}</b> nasabah<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_standard_layout(
        fig,
        f"Confusion Matrix — {model_name} ({scenario_label})",
        height=360,
    )


def plot_combined_roc_curve(models_dict: Dict[str, Any], scenario_title: str) -> go.Figure:
    """
    Membuat grafik kurva ROC gabungan untuk keempat model pada satu skenario,
    lengkap dengan nilai AUC di legenda dan garis acak (0.5).
    """
    palette = {
        "Gradient Boosting": COLORS["navy"],
        "Random Forest": COLORS["teal"],
        "K-Nearest Neighbors": COLORS["gold"],
        "Decision Tree": COLORS["danger"],
    }

    fig = go.Figure()

    for model_name, m_data in models_dict.items():
        fpr = m_data["fpr"]
        tpr = m_data["tpr"]
        auc_val = m_data["roc_auc"]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc_val:.4f})",
                line=dict(color=palette.get(model_name, COLORS["blue"]), width=2.8),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "False Positive Rate: %{x:.3f}<br>"
                    "True Positive Rate: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Garis acak 0.5
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Tebakan Acak (AUC = 0.5000)",
            line=dict(color="#9CA3AF", width=1.8, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_xaxes(title_text="False Positive Rate (1 - Spesifisitas)", range=[0, 1], gridcolor="#F3F4F6")
    fig.update_yaxes(title_text="True Positive Rate (Recall / Sensitivitas)", range=[0, 1.02], gridcolor="#F3F4F6")
    fig = _apply_standard_layout(fig, f"Kurva ROC Gabungan Keempat Model — {scenario_title}", height=450)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        )
    )
    return fig


def plot_feature_importance(fi_dict: Dict[str, float], title: str) -> go.Figure:
    """Membuat bar chart horizontal untuk Feature Importance dari Random Forest."""
    df_fi = (
        pd.DataFrame(
            {
                "Fitur": list(fi_dict.keys()),
                "Kepentingan": list(fi_dict.values()),
            }
        )
        .sort_values("Kepentingan", ascending=True)
        .reset_index(drop=True)
    )
    df_fi["Persentase (%)"] = df_fi["Kepentingan"] * 100.0
    df_fi["Label"] = df_fi["Persentase (%)"].apply(lambda v: f"{v:.2f}%")

    colors = [
        COLORS["danger"] if idx == len(df_fi) - 1 else COLORS["teal"] if idx == len(df_fi) - 2 else COLORS["navy"]
        for idx in range(len(df_fi))
    ]

    fig = go.Figure(
        go.Bar(
            x=df_fi["Persentase (%)"],
            y=df_fi["Fitur"],
            orientation="h",
            text=df_fi["Label"],
            textposition="outside",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Kontribusi Kepentingan: <b>%{x:.2f}%</b><extra></extra>",
        )
    )
    max_val = float(df_fi["Persentase (%)"].max()) if len(df_fi) > 0 else 50.0
    fig.update_xaxes(title_text="Tingkat Kepentingan Fitur (%)", range=[0, max_val * 1.18], gridcolor="#F3F4F6")
    fig.update_yaxes(title_text="Nama Fitur", showgrid=False)
    return _apply_standard_layout(fig, title, height=390)


def plot_churn_probability_gauge(prob_churn: float) -> go.Figure:
    """
    Membuat Gauge Chart untuk menampilkan probabilitas churn nasabah (0% - 100%)
    dengan zona risiko Rendah (<35%), Sedang (35%-65%), dan Tinggi (>65%).
    """
    pct = prob_churn * 100.0
    if pct < 35.0:
        bar_color = COLORS["teal"]
    elif pct < 65.0:
        bar_color = COLORS["gold"]
    else:
        bar_color = COLORS["danger"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "valueformat": ".2f", "font": {"size": 36, "color": COLORS["navy"]}},
            title={"text": "<b>Probabilitas Nasabah Churn</b>", "font": {"size": 15, "color": COLORS["navy"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["navy"]},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#E5E7EB",
                "steps": [
                    {"range": [0, 35], "color": "rgba(27, 153, 139, 0.16)"},
                    {"range": [35, 65], "color": "rgba(244, 185, 66, 0.22)"},
                    {"range": [65, 100], "color": "rgba(228, 87, 46, 0.20)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["danger"], "width": 4},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
        height=310,
        margin=dict(l=30, r=30, t=55, b=20),
    )
    return fig
