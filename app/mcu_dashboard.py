# ====================================================
# mcu_dashboard.py
# Double-click file ini → langsung buka dashboard cakep di browser
# Dependencies: pip install dash plotly pandas scikit-learn
# ====================================================

import os
os.environ.setdefault('OMP_NUM_THREADS', '8')  # menghindari memory leak MKL

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ====================================================
# 1. Load dataset
# ====================================================
df = pd.read_csv("Cleaned_agg_pasien_MCU_2.csv")

# ====================================================
# 2. Quick preprocessing
# ====================================================
df_cluster = df.drop(columns=['BADGE'], errors='ignore').copy()

# BMI
if 'TINGGI' in df_cluster.columns and 'BERAT' in df_cluster.columns:
    df_cluster['BMI'] = df_cluster['BERAT'] / ((df_cluster['TINGGI'] / 100) ** 2)

# MAP
if 'SISTOLIK' in df_cluster.columns and 'DIASTOLIK' in df_cluster.columns:
    df_cluster['MAP'] = df_cluster['DIASTOLIK'] + (df_cluster['SISTOLIK'] - df_cluster['DIASTOLIK']) / 3

# ====================================================
# 3. Scaling → Clustering → t-SNE
# ====================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(df_cluster.fillna(df_cluster.median()))

kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
labels = kmeans.fit_predict(scaled)

tsne = TSNE(
    n_components=3,
    perplexity=30,
    random_state=42,
    init='pca',
    learning_rate='auto',
    max_iter=1000
)
tsne_3d = tsne.fit_transform(scaled)

# ====================================================
# 4. Combine results
# ====================================================
result = pd.DataFrame({
    "BADGE": df["BADGE"].astype(str),
    "Cluster": labels.astype(str),
    "tSNE_1": tsne_3d[:, 0],
    "tSNE_2": tsne_3d[:, 1],
    "tSNE_3": tsne_3d[:, 2],
    "BMI": df_cluster.get("BMI", np.nan).round(1),
    "GULA_DARAH_PUASA": df.get("GULA_DARAH_PUASA", np.nan),
    "TRIGLISERIDA": df.get("TRIGLISERIDA", np.nan),
    "HB": df.get("HB", np.nan),
    "UMUR": df.get("UMUR", np.nan)
})

# ====================================================
# 5. DASH APP
# ====================================================
app = dash.Dash(__name__, title="MCU Patient Clustering")

app.layout = html.Div(
    children=[
        # ======================= Title =======================
        html.H1(
            "Hasil Clustering Pasien MCU",
            style={
                "textAlign": "center",
                "color": "#2ecc71",
                "fontWeight": "700",
                "letterSpacing": "1px"
            }
        ),

        html.H4(
            "K-Means + t-SNE 3D • 5 Cluster • Rotatable Visualization",
            style={
                "textAlign": "center",
                "color": "#e67e22",
                "marginBottom": "40px"
            }
        ),

        # ======================= 3D Plot Card =======================
        html.Div(
            children=[
                dcc.Graph(id="3d-plot", style={"height": "700px"})
            ],
            style={
                "padding": "16px",
                "border": "1px solid #1f2937",
                "borderRadius": "16px",
                "backgroundColor": "#111418",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.35)"
            }
        ),

        html.Hr(style={"borderColor": "#2ecc71", "marginTop": "40px"}),

        # ======================= Search Section =======================
        html.Div(
            children=[
                html.H3("Cari Pasien berdasarkan BADGE", style={"color": "#2ecc71"}),

                dcc.Input(
                    id="search-badge",
                    type="text",
                    placeholder="Ketik BADGE...",
                    style={
                        "width": "280px",
                        "padding": "10px 14px",
                        "marginTop": "10px",
                        "backgroundColor": "#0f1115",
                        "border": "1px solid #2a2f3a",
                        "borderRadius": "12px",
                        "color": "#e6edf3",
                        "fontSize": "15px",
                        "transition": "all 0.2s ease"
                    }
                ),

                html.Div(id="search-result", style={"marginTop": "30px"})
            ],
            style={
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": "#161b22",
                "border": "1px solid #1f2937",
                "borderRadius": "12px",
                "marginTop": "40px",
                "boxShadow": "0 0 12px rgba(230, 126, 34, 0.15)"
            }
        )
    ],
    style={
        "textAlign": "center",
        "padding": "24px",
        "backgroundColor": "#111418",
        "border": "1px solid #1f2937",
        "borderRadius": "16px",
        "marginTop": "40px",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.35)"
    }
)

# ====================================================
# CALLBACK 1 — 3D Plot
# ====================================================
@app.callback(
    Output("3d-plot", "figure"),
    Input("3d-plot", "id")
)
def update_plot(_):
    fig = px.scatter_3d(
        result,
        x="tSNE_1",
        y="tSNE_2",
        z="tSNE_3",
        color="Cluster",
        hover_data=["BADGE", "BMI", "GULA_DARAH_PUASA", "TRIGLISERIDA", "HB"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.85, line=dict(width=0.5, color="#e67e22"))
    )

    fig.update_layout(
        scene_aspectmode="cube",
        height=700,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22"
    )

    return fig

# ====================================================
# CALLBACK 2 — Search BADGE
# ====================================================
@app.callback(
    Output("search-result", "children"),
    Input("search-badge", "value")
)
def search_patient(badge):
    if not badge:
        return ""

    found = result[result["BADGE"].str.contains(badge, case=False, na=False)]
    if found.empty:
        return html.Div("Tidak ditemukan", style={"color": "red"})

    return html.Table(
        children=[
            html.Thead(
                html.Tr([
                    html.Th(col, style={'padding': '8px', 'border': '1px solid #2ecc71'})
                    for col in found.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(
                        found.iloc[i][col],
                        style={
                            "padding": "6px 10px",
                            "border": "1px solid #444",
                            "backgroundColor": "#161b22" if i % 2 == 0 else "#1f2937"
                        }
                    ) for col in found.columns
                ]) for i in range(len(found))
            ])
        ],
        style={
            "margin": "auto",
            "borderCollapse": "collapse",
            "color": "#e6edf3",
            "width": "90%",
            "fontSize": "14px",
            "boxShadow": "0 0 12px rgba(46, 204, 113, 0.25)"
        }
    )

# ====================================================
# 6. Run app
# ====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DASHBOARD SEDANG JALAN!")
    print("Tunggu 3–5 detik, browser akan terbuka otomatis")
    print("Jika tidak → buka manual: http://127.0.0.1:8050")
    print("=" * 60)

    app.run(host="127.0.0.1", port=8050, debug=False)
