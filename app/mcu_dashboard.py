# ====================================================
# mcu_dashboard.py - Neovim Inspired Theme
# Double-click file ini → langsung buka dashboard cakep di browser
# Dependencies: pip install dash plotly pandas scikit-learn
# ====================================================

import os
os.environ.setdefault('OMP_NUM_THREADS', '8')

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ====================================================
# NEOVIM COLOR PALETTE
# ====================================================
NVIM = {
    'bg': '#1e222a',           # background
    'bgAlt': '#282c34',       # lighter bg
    'fg': '#abb2bf',           # foreground text
    'selection': '#3e4451',    # selection
    'comment': '#5c6370',      # comments
    'red': '#e06c75',          # error/red
    'orange': '#d19a66',       # orange
    'yellow': '#e5c07b',       # warning/yellow
    'green': '#98c379',        # success/green
    'cyan': '#56b6c2',         # cyan
    'blue': '#61afef',         # info/blue
    'purple': '#c678dd',       # purple
    'gutter': '#4b5263',       # line numbers
    'border': '#3e4451',       # borders
    'statusline': '#2c313c',   # statusline bg
}

# ====================================================
# 1. Load dataset
# ====================================================
df = pd.read_csv("Cleaned_agg_pasien_MCU_2.csv")

# ====================================================
# 2. Quick preprocessing
# ====================================================
df_cluster = df.drop(columns=['BADGE'], errors='ignore').copy()

if 'TINGGI' in df_cluster.columns and 'BERAT' in df_cluster.columns:
    df_cluster['BMI'] = df_cluster['BERAT'] / ((df_cluster['TINGGI'] / 100) ** 2)

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
# 5. DASH APP - NEOVIM STYLED
# ====================================================
app = dash.Dash(__name__, title="MCU Patient Clustering")

# Force dark mode everywhere
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%css%}
        <style>
            /* Global reset biar gak ada white outline */
            * {
                outline: none !important;
            }
            body, html {
                background: #1e222a !important;
            }
            table, th, td {
                border-color: #3e4451 !important;
            }
            /* Only override container, not the SVG canvas */
            .js-plotly-plot, .plot-container {
                background: #1e222a !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""



app.layout = html.Div(
    children=[
        # ======================= Header =======================
        html.Div(
            children=[
                html.Div(
                    "// MCU_DASHBOARD.PY",
                    style={
                        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                        "fontSize": "12px",
                        "color": NVIM["comment"],
                        "marginBottom": "8px",
                        "letterSpacing": "0.5px"
                    }
                ),
                html.H1(
                    "fn cluster_patients()",
                    style={
                        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                        "color": NVIM["purple"],
                        "fontWeight": "600",
                        "fontSize": "32px",
                        "margin": "0",
                        "letterSpacing": "-0.5px"
                    }
                ),
                html.Div(
                    children=[
                        html.Span("→ ", style={"color": NVIM["cyan"]}),
                        html.Span("K-Means ", style={"color": NVIM["green"]}),
                        html.Span("+ ", style={"color": NVIM["fg"]}),
                        html.Span("t-SNE ", style={"color": NVIM["blue"]}),
                        html.Span("+ ", style={"color": NVIM["fg"]}),
                        html.Span("3D Viz", style={"color": NVIM["orange"]})
                    ],
                    style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "14px",
                        "marginTop": "12px"
                    }
                )
            ],
            style={
                "textAlign": "center",
                "padding": "24px",
                "backgroundColor": NVIM["statusline"],
                "borderBottom": f"2px solid {NVIM['blue']}",
                "marginBottom": "24px"
            }
        ),

        # ======================= 3D Plot =======================
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Span("1 ", style={"color": NVIM["gutter"], "marginRight": "12px"}),
                        html.Span("let", style={"color": NVIM["purple"], "marginRight": "8px"}),
                        html.Span("visualization", style={"color": NVIM["fg"], "marginRight": "8px"}),
                        html.Span("=", style={"color": NVIM["cyan"], "marginRight": "8px"}),
                        html.Span("{", style={"color": NVIM["fg"]})
                    ],
                    style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "13px",
                        "padding": "8px 16px",
                        "backgroundColor": NVIM["bgAlt"],
                        "borderBottom": f"1px solid {NVIM['border']}",
                        "textAlign": "left"
                    }
                ),
                dcc.Graph(id="3d-plot", style={"height": "650px"})
            ],
            style={
                "border": f"1px solid {NVIM['border']}",
                "borderRadius": "4px",
                "backgroundColor": NVIM["bg"],
                "overflow": "hidden",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.3)"
            }
        ),

        # ======================= Divider =======================
        html.Div(
            "// ────────────────────────────────────────────────────────────────",
            style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "11px",
                "color": NVIM["comment"],
                "textAlign": "center",
                "margin": "32px 0"
            }
        ),

        # ======================= Search Section =======================
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Span("function ", style={"color": NVIM["purple"], "marginRight": "6px"}),
                        html.Span("search", style={"color": NVIM["yellow"], "marginRight": "4px"}),
                        html.Span("(", style={"color": NVIM["fg"]}),
                        html.Span("badge", style={"color": NVIM["orange"], "fontStyle": "italic"}),
                        html.Span(":", style={"color": NVIM["fg"], "marginRight": "4px"}),
                        html.Span("string", style={"color": NVIM["cyan"]}),
                        html.Span(") {", style={"color": NVIM["fg"]})
                    ],
                    style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "14px",
                        "marginBottom": "16px",
                        "textAlign": "left"
                    }
                ),

                html.Div(
                    children=[
                        html.Span("❯ ", style={"color": NVIM["green"], "marginRight": "8px"}),
                        dcc.Input(
                            id="search-badge",
                            type="text",
                            placeholder="Ketik BADGE...",
                            style={
                                "fontFamily": "'JetBrains Mono', monospace",
                                "width": "300px",
                                "padding": "10px 14px",
                                "backgroundColor": NVIM["bgAlt"],
                                "border": f"1px solid {NVIM['border']}",
                                "borderRadius": "2px",
                                "color": NVIM["fg"],
                                "fontSize": "13px",
                                "outline": "none"
                            }
                        )
                    ],
                    style={"textAlign": "left", "marginBottom": "20px"}
                ),

                html.Div(id="search-result", style={"marginTop": "20px"})
            ],
            style={
                "padding": "20px 24px",
                "backgroundColor": NVIM["bg"],
                "border": f"1px solid {NVIM['border']}",
                "borderRadius": "4px",
                "marginTop": "24px",
                "textAlign": "left"
            }
        ),

        # ======================= Footer =======================
        html.Div(
            children=[
                html.Span("}", style={"color": NVIM["fg"], "marginRight": "16px"}),
                html.Span("// end of file", style={"color": NVIM["comment"]})
            ],
            style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "12px",
                "textAlign": "center",
                "marginTop": "32px",
                "paddingBottom": "16px"
            }
        )
    ],
    style={
        "padding": "32px",
        "backgroundColor": NVIM["bg"],
        "minHeight": "100vh",
        "color": NVIM["fg"],
        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace"
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
        color_discrete_sequence=[NVIM['red'], NVIM['green'], NVIM['yellow'], 
                                 NVIM['blue'], NVIM['purple']],
        template="plotly_dark"
    )

    fig.update_traces(
        marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color=NVIM['cyan']))
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor=NVIM['bg'],
                gridcolor=NVIM['border'],
                showbackground=True,
                zerolinecolor=NVIM['gutter']
            ),
            yaxis=dict(
                backgroundcolor=NVIM['bg'],
                gridcolor=NVIM['border'],
                showbackground=True,
                zerolinecolor=NVIM['gutter']
            ),
            zaxis=dict(
                backgroundcolor=NVIM['bg'],
                gridcolor=NVIM['border'],
                showbackground=True,
                zerolinecolor=NVIM['gutter']
            )
        ),
        scene_aspectmode="cube",
        height=650,
        paper_bgcolor=NVIM['bg'],
        plot_bgcolor=NVIM['bg'],
        font=dict(family="'JetBrains Mono', monospace", color=NVIM['fg'], size=11),
        margin=dict(l=0, r=0, t=0, b=0)
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
        return html.Div(
            "// no results",
            style={
                "color": NVIM['comment'],
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "12px"
            }
        )

    found = result[result["BADGE"].str.contains(badge, case=False, na=False)]
    
    if found.empty:
        return html.Div(
            children=[
                html.Span("✗ ", style={"color": NVIM['red'], "marginRight": "8px"}),
                html.Span("Error: Badge not found", style={"color": NVIM['red']})
            ],
            style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize": "13px"
            }
        )

    return html.Table(
        children=[
            html.Thead(
                html.Tr([
                    html.Th(
                        col,
                        style={
                            'padding': '10px 14px',
                            'border': f'1px solid {NVIM["border"]}',
                            'backgroundColor': NVIM['statusline'],
                            'color': NVIM['blue'],
                            'fontWeight': '600',
                            'fontSize': '12px',
                            'textAlign': 'left'
                        }
                    ) for col in found.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(
                        found.iloc[i][col],
                        style={
                            "padding": "8px 14px",
                            "border": f"1px solid {NVIM['border']}",
                            "backgroundColor": NVIM['bg_alt'] if i % 2 == 0 else NVIM['bg'],
                            "color": NVIM['fg'],
                            "fontSize": "12px"
                        }
                    ) for col in found.columns
                ]) for i in range(len(found))
            ])
        ],
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "fontFamily": "'JetBrains Mono', monospace",
            "marginTop": "12px"
        }
    )

# ====================================================
# 6. Run app
# ====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("// NEOVIM DASHBOARD ACTIVE")
    print("// Server running on http://127.0.0.1:8050")
    print("// Press Ctrl+C to exit")
    print("=" * 60)

    app.run(host="127.0.0.1", port=8050, debug=False)