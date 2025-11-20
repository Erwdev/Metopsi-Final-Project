import os
# Set OMP_NUM_THREADS to avoid excessive CPU usage warnings
os.environ.setdefault('OMP_NUM_THREADS', '8')

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ====================================================
# NEOVIM COLOR PALETTE
# ====================================================
NVIM = {
    'bg': '#1e222a',        # background
    'bgAlt': '#282c34',     # lighter bg
    'fg': '#abb2bf',        # foreground text
    'selection': '#3e4451', # selection
    'comment': '#5c6370',   # comments
    'red': '#e06c75',       # error/red
    'orange': '#d19a66',    # orange
    'yellow': '#e5c07b',    # warning/yellow
    'green': '#98c379',     # success/green
    'cyan': '#56b6c2',      # cyan
    'blue': '#61afef',      # info/blue
    'purple': '#c678dd',    # purple
    'gutter': '#4b5263',    # line numbers
    'border': '#3e4451',    # borders
    'statusline': '#2c313c',# statusline bg
}

CLUSTER_COLORS_LIST = [NVIM['red'], NVIM['green'], NVIM['yellow'], NVIM['blue'], NVIM['purple']]
K = 5
COLOR_MAP = {str(i): CLUSTER_COLORS_LIST[i] for i in range(K)}

# ====================================================
# 1. Load dataset & Feature Selection
# ====================================================
CLUSTERING_COLS = [
    'TINGGI', 'BERAT', 'NADI', 'HB', 'LEUKOSIT', 'TROMBOSIT', 
    'BILIRUBIN_TOTAL', 'SGPT', 'SGOT', 'KOLEST_TOTAL', 'TRIGLISERIDA', 
    'HDL_KOLEST', 'LDL_KOLEST', 'UREUM', 'KREATININ', 'ASAM_URAT_GINJAL', 
    'GULA_DARAH_PUASA'
]

PROFILE_COLS = [
    'BMI_CALC', 'GULA_DARAH_PUASA', 'KOLEST_TOTAL', 'TRIGLISERIDA', 
    'ASAM_URAT_GINJAL', 'UREUM', 'KREATININ', 'SGPT', 'HB'
]

try:
    df = pd.read_csv("Cleaned_agg_pasien_MCU_2.csv")
    for col in CLUSTERING_COLS:
        if col not in df.columns: df[col] = 0
            
except FileNotFoundError:
    print("Generating Dummy Data...")
    N = 150
    data = {
        'BADGE': [f'ID{i:03}' for i in range(N)],
        'TINGGI': np.random.normal(1650, 100, N),
        'BERAT': np.random.normal(650, 150, N),
        'NADI': np.random.normal(80, 10, N),
        'HB': np.random.normal(14000, 2000, N),
        'LEUKOSIT': np.random.normal(7000, 2000, N),
        'TROMBOSIT': np.random.normal(2500, 500, N),
        'BILIRUBIN_TOTAL': np.random.normal(1000, 200, N),
        'SGPT': np.random.normal(250, 100, N),
        'SGOT': np.random.normal(250, 100, N),
        'KOLEST_TOTAL': np.random.normal(1800, 400, N),
        'TRIGLISERIDA': np.random.normal(1500, 500, N),
        'HDL_KOLEST': np.random.normal(450, 100, N),
        'LDL_KOLEST': np.random.normal(1100, 300, N),
        'UREUM': np.random.normal(250, 50, N),
        'KREATININ': np.random.normal(90, 20, N),
        'ASAM_URAT_GINJAL': np.random.normal(5000, 1000, N),
        'GULA_DARAH_PUASA': np.random.normal(1000, 300, N)
    }
    df = pd.DataFrame(data)

# ====================================================
# 2. Preprocessing
# ====================================================
df_cluster = df.copy()
try:
    df_cluster['BMI_CALC'] = (df_cluster['BERAT'] / 10) / ((df_cluster['TINGGI'] / 1000) ** 2)
except KeyError:
    df_cluster['BMI_CALC'] = 0

df_filled = df_cluster[CLUSTERING_COLS].fillna(df_cluster[CLUSTERING_COLS].median())

# ====================================================
# 3. Modeling
# ====================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(df_filled)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
labels = kmeans.fit_predict(scaled)
sil_score = silhouette_score(scaled, labels)

tsne = TSNE(n_components=3, perplexity=min(30, len(df_filled) - 1), random_state=42, init='pca', learning_rate='auto', max_iter=1000)
tsne_3d = tsne.fit_transform(scaled)

# ====================================================
# 4. Data Preparation
# ====================================================
result = df.copy()
result['Cluster'] = labels.astype(str)
result['tSNE_1'] = tsne_3d[:, 0]
result['tSNE_2'] = tsne_3d[:, 1]
result['tSNE_3'] = tsne_3d[:, 2]
result['BMI_CALC'] = df_cluster['BMI_CALC'].round(1)

# Trajectory
centroids_df = result.groupby('Cluster')[['tSNE_1', 'tSNE_2', 'tSNE_3']].mean().reset_index()
final_centroids_tsne_data = centroids_df[['tSNE_1', 'tSNE_2', 'tSNE_3']].values
origin_tsne = tsne_3d.mean(axis=0)
steps = 10
trajectory_data = []
for i in range(K):
    cluster_label = str(i)
    c_tsne = final_centroids_tsne_data[i] 
    for step in range(steps + 1):
        frac = step / steps
        t1 = origin_tsne[0] + (c_tsne[0] - origin_tsne[0]) * frac
        t2 = origin_tsne[1] + (c_tsne[1] - origin_tsne[1]) * frac
        t3 = origin_tsne[2] + (c_tsne[2] - origin_tsne[2]) * frac
        trajectory_data.append({"Cluster": cluster_label, "tSNE_1": t1, "tSNE_2": t2, "tSNE_3": t3, "Step": step})
trajectory_df = pd.DataFrame(trajectory_data)

# Radar Data
valid_profile_cols = [c for c in PROFILE_COLS if c in result.columns]
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(result[valid_profile_cols].fillna(0)), columns=valid_profile_cols)
df_normalized['Cluster'] = result['Cluster']
radar_data = df_normalized.groupby('Cluster').mean().reset_index()

feature_options = [{'label': col, 'value': col} for col in valid_profile_cols]

# ====================================================
# 5. DASH APP (BENTO GRID LAYOUT)
# ====================================================
app = dash.Dash(__name__, title="MCU Medical Clustering")

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            * {{ outline: none !important; box-sizing: border-box; }}
            body, html {{ background: {NVIM['bg']} !important; margin: 0; padding: 0; color: {NVIM['fg']}; }}
            .js-plotly-plot, .plot-container {{ background: {NVIM['bg']} !important; }}
            
            /* BENTO GRID SYSTEM */
            .bento-container {{
                display: grid;
                grid-template-columns: 2.5fr 1.5fr; /* Kiri (3D) lebih besar dari Kanan */
                grid-template-rows: auto auto;
                gap: 20px;
                padding: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }}
            
            .bento-col-left {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            
            .bento-col-right {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            
            .bento-card {{
                background-color: {NVIM['bg']};
                border: 1px solid {NVIM['border']};
                border-radius: 8px;
                padding: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            
            .card-header {{
                background-color: {NVIM['bgAlt']};
                padding: 10px 16px;
                border-bottom: 1px solid {NVIM['border']};
                font-family: 'JetBrains Mono';
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .card-body {{
                padding: 16px;
                flex-grow: 1;
            }}

            /* Responsive for Mobile */
            @media (max-width: 1200px) {{
                .bento-container {{ grid-template-columns: 1fr; }}
            }}

            /* Slider Styling */
            .rc-slider-track {{ background-color: {NVIM['blue']} !important; }}
            .rc-slider-handle {{ border-color: {NVIM['blue']} !important; background-color: {NVIM['bgAlt']} !important; }}
            .rc-slider-mark-text {{ color: {NVIM['fg']} !important; font-family: 'JetBrains Mono'; }}
        </style>
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>
"""

app.layout = html.Div([
    # --- HEADER ---
    html.Div([
        html.Div([
            html.Div("// MCU_ANALYSIS_DASHBOARD", style={"fontFamily": "'JetBrains Mono'", "fontSize": "12px", "color": NVIM["comment"]}),
            html.H1("fn medical_clustering()", style={"fontFamily": "'JetBrains Mono'", "color": NVIM["purple"], "fontSize": "24px", "margin": "4px 0"}),
        ]),
        html.Div([
            html.Div([
                html.Span("Silhouette Score: ", style={"color": NVIM['fg']}),
                html.Span(f"{sil_score:.3f}", style={"color": NVIM['green'] if sil_score > 0.25 else NVIM['red'], "fontWeight": "bold"}),
            ], style={"fontFamily": "'JetBrains Mono'", "fontSize": "14px"}),
            html.Div([
                html.Span("N_Patients: ", style={"color": NVIM['fg']}),
                html.Span(f"{len(df)}", style={"color": NVIM['blue']}),
            ], style={"fontFamily": "'JetBrains Mono'", "fontSize": "14px"})
        ], style={"textAlign": "right", "borderLeft": f"2px solid {NVIM['selection']}", "paddingLeft": "16px"})
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "16px 24px", "backgroundColor": NVIM["statusline"], "borderBottom": f"2px solid {NVIM['blue']}"}),

    # --- BENTO GRID ---
    html.Div(className="bento-container", children=[
        
        # LEFT COLUMN (Main Visual & Search)
        html.Div(className="bento-col-left", children=[
            
            # CARD 1: 3D PROJECTION
            html.Div(className="bento-card", style={"minHeight": "650px"}, children=[
                html.Div(className="card-header", children=[
                    html.Div([html.Span("let ", style={"color": NVIM['purple']}), html.Span("manifold_3d", style={"color": NVIM['fg']}), html.Span(" = ", style={"color": NVIM['cyan']}), html.Span("tSNE(Hematology)", style={"color": NVIM['yellow']})]),
                    html.Span("[Interactive]", style={"color": NVIM['comment'], "fontSize": "11px"})
                ]),
                html.Div(className="card-body", children=[
                    # Graph
                    dcc.Graph(id="3d-plot", style={"height": "550px"}, config={'displayModeBar': True}),
                    
                    # Slider Control
                    html.Div([
                        html.Label([html.Span("sim_step: ", style={"color": NVIM["purple"]}), html.Span("0", id="slider-output", style={"color": NVIM["orange"]})], style={"fontFamily": "'JetBrains Mono'", "marginBottom": "10px", "fontSize": "12px"}),
                        dcc.Slider(
                            id='step-slider', min=0, max=steps, step=1, value=steps, 
                            marks={i: {'label': str(i), 'style': {'color': NVIM['fg']}} for i in range(steps + 1)}
                        )
                    ], style={"paddingTop": "10px"})
                ])
            ]),

            # CARD 2: SEARCH (Bottom Left)
            html.Div(className="bento-card", children=[
                html.Div(className="card-header", children=[
                    html.Div([html.Span("fn ", style={"color": NVIM['purple']}), html.Span("search_patient", style={"color": NVIM['blue']}), html.Span("(badge_id)", style={"color": NVIM['orange']})]),
                ]),
                html.Div(className="card-body", children=[
                    dcc.Input(
                        id="search-badge", type="text", placeholder="Ketik BADGE ID... (Tekan Enter)", debounce=True,
                        style={"fontFamily": "'JetBrains Mono'", "width": "100%", "padding": "12px", "backgroundColor": NVIM["bgAlt"], "border": f"1px solid {NVIM['border']}", "color": NVIM["fg"], "outline": "none", "borderRadius": "4px"}
                    ),
                    dcc.Loading(
                        id="loading-search", type="dot", color=NVIM['purple'],
                        children=html.Div(id="search-result", style={"marginTop": "20px", "overflowX": "auto"})
                    )
                ])
            ])
        ]),

        # RIGHT COLUMN (Profiling)
        html.Div(className="bento-col-right", children=[
            
            # CARD 3: RADAR CHART
            html.Div(className="bento-card", style={"flex": "1"}, children=[
                html.Div(className="card-header", children=[
                    html.Div([html.Span("struct ", style={"color": NVIM['purple']}), html.Span("ClusterProfile", style={"color": NVIM['yellow']})]),
                ]),
                html.Div(className="card-body", children=[
                     dcc.Graph(id="radar-plot", style={"height": "350px"}, config={'displayModeBar': False})
                ])
            ]),

            # CARD 4: BOX PLOT
            html.Div(className="bento-card", style={"flex": "1"}, children=[
                html.Div(className="card-header", children=[
                    html.Div([html.Span("fn ", style={"color": NVIM['purple']}), html.Span("distribution", style={"color": NVIM['blue']}), html.Span("()", style={"color": NVIM['fg']})]),
                    # Dropdown inside header for compact look
                    dcc.Dropdown(
                        id='feature-dropdown', options=feature_options, value='GULA_DARAH_PUASA', clearable=False,
                        style={'width': '180px', 'fontSize': '12px', 'color': '#000'}
                    ),
                ]),
                html.Div(className="card-body", children=[
                    dcc.Graph(id="box-plot", style={"height": "350px"}, config={'displayModeBar': False})
                ])
            ])
        ])
    ]),
    
    html.Div([html.Span("}", style={"color": NVIM["fg"]}), html.Span("// end of analysis", style={"color": NVIM["comment"], "marginLeft": "12px"})], style={"fontFamily": "'JetBrains Mono'", "textAlign": "center", "padding": "20px"})

], style={"backgroundColor": NVIM["bg"], "minHeight": "100vh"})

# ====================================================
# CALLBACKS
# ====================================================
@app.callback(Output("slider-output", "children"), Input("step-slider", "value"))
def update_label(val): return str(val)

# 1. UPDATE 3D PLOT
@app.callback(Output("3d-plot", "figure"), Input("step-slider", "value"))
def update_3d(selected_step):
    # Default step jika None (misal saat inisialisasi)
    if selected_step is None: selected_step = steps
        
    custom_hover = []
    for _, row in result.iterrows():
        txt = (f"<b>BADGE:</b> {row['BADGE']}<br>"
               f"Cluster: {row['Cluster']}<br>"
               f"BMI: {row.get('BMI_CALC','-')}<br>"
               f"Gula: {row.get('GULA_DARAH_PUASA','-')}<br>"
               f"Asam Urat: {row.get('ASAM_URAT_GINJAL','-')}")
        custom_hover.append(txt)

    fig = go.Figure()
    
    # Trace 1: Pasien (Selalu muncul)
    fig.add_trace(go.Scatter3d(
        x=result['tSNE_1'], y=result['tSNE_2'], z=result['tSNE_3'], mode='markers', name='Pasien',
        marker=dict(size=4, opacity=0.6, color=[COLOR_MAP[c] for c in result['Cluster']], line=dict(width=0.5, color=NVIM['selection'])),
        hoverinfo='text', hovertext=custom_hover
    ))
    
    # Trace 2: Centroid & Trajectory (Bergantung Step)
    if not trajectory_df.empty:
        c_step = trajectory_df[trajectory_df["Step"] == selected_step]
        c_traj = trajectory_df[trajectory_df["Step"] <= selected_step]
        
        for i in range(K):
            cl = str(i)
            ct = c_traj[c_traj["Cluster"] == cl]
            # Garis jejak
            fig.add_trace(go.Scatter3d(x=ct['tSNE_1'], y=ct['tSNE_2'], z=ct['tSNE_3'], mode='lines', showlegend=False, line=dict(color=COLOR_MAP[cl], width=4, dash='solid'), hoverinfo='none'))
        
        # Marker Centroid
        fig.add_trace(go.Scatter3d(x=c_step['tSNE_1'], y=c_step['tSNE_2'], z=c_step['tSNE_3'], mode='markers', name='Centroids', marker=dict(size=12, symbol='diamond', color=[COLOR_MAP[c] for c in c_step['Cluster']], line=dict(width=2, color=NVIM['fg'])), hovertext=[f"Centroid {c}" for c in c_step['Cluster']]))

    # PERBAIKAN UTAMA: Menghapus titlefont dan mengganti dengan struktur dict nested
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor=NVIM['bgAlt'], 
                gridcolor=NVIM['border'], 
                title=dict(text='tSNE-1', font=dict(color=NVIM['cyan']))
            ),
            yaxis=dict(
                backgroundcolor=NVIM['bgAlt'], 
                gridcolor=NVIM['border'], 
                title=dict(text='tSNE-2', font=dict(color=NVIM['cyan']))
            ),
            zaxis=dict(
                backgroundcolor=NVIM['bgAlt'], 
                gridcolor=NVIM['border'], 
                title=dict(text='tSNE-3', font=dict(color=NVIM['cyan']))
            ),
            aspectmode="cube"
        ),
        paper_bgcolor=NVIM['bg'], plot_bgcolor=NVIM['bg'], font=dict(family="'JetBrains Mono'", color=NVIM['fg']),
        margin=dict(l=0, r=0, t=0, b=0), legend=dict(bgcolor=NVIM['bgAlt'], bordercolor=NVIM['border'])
    )
    return fig

# 2. UPDATE RADAR CHART
@app.callback(Output("radar-plot", "figure"), Input("step-slider", "value"))
def update_radar(_):
    fig = go.Figure()
    categories = valid_profile_cols
    
    for i in range(K):
        cluster_idx = str(i)
        if cluster_idx in radar_data['Cluster'].values:
            values = radar_data[radar_data['Cluster'] == cluster_idx][categories].values.flatten().tolist()
            values += values[:1] 
            cats_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values, theta=cats_closed, fill='toself', name=f'Cluster {cluster_idx}',
                line=dict(color=COLOR_MAP[cluster_idx]), opacity=0.5
            ))

    fig.update_layout(
        polar=dict(
            bgcolor=NVIM['bg'],
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=NVIM['border']),
            angularaxis=dict(tickfont=dict(color=NVIM['fg'], size=10), gridcolor=NVIM['border'], rotation=90)
        ),
        paper_bgcolor=NVIM['bg'], font=dict(family="'JetBrains Mono'", color=NVIM['fg']),
        margin=dict(l=40, r=40, t=20, b=20),
        legend=dict(bgcolor=NVIM['bgAlt'], bordercolor=NVIM['border'], font=dict(size=10))
    )
    return fig

# 3. UPDATE BOX PLOT
@app.callback(Output("box-plot", "figure"), Input("feature-dropdown", "value"))
def update_box(feature):
    fig = px.box(result, x="Cluster", y=feature, color="Cluster", color_discrete_map=COLOR_MAP, template="plotly_dark")
    fig.update_layout(
        paper_bgcolor=NVIM['bg'], plot_bgcolor=NVIM['bg'],
        font=dict(family="'JetBrains Mono'", color=NVIM['fg']),
        xaxis=dict(gridcolor=NVIM['border']), yaxis=dict(gridcolor=NVIM['border']),
        title=f"{feature}", title_font_size=12,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# 4. SEARCH
@app.callback(Output("search-result", "children"), Input("search-badge", "value"))
def search(badge):
    if not badge: return html.Div("// waiting for input...", style={"color": NVIM['comment'], "fontFamily": "'JetBrains Mono'"})
    found = result[result["BADGE"].str.contains(badge, case=False, na=False)]
    if found.empty: return html.Div("Badge ID not found.", style={"color": NVIM['red'], "fontFamily": "'JetBrains Mono'"})
    
    display_cols = ['BADGE', 'Cluster', 'BMI_CALC', 'GULA_DARAH_PUASA', 'KOLEST_TOTAL', 'HB', 'UREUM', 'KREATININ']
    final_cols = [c for c in display_cols if c in found.columns]
    
    return html.Table([
        html.Thead(html.Tr([html.Th(c, style={'padding':'10px', 'border':f'1px solid {NVIM["border"]}', 'color':NVIM['blue'], 'backgroundColor':NVIM['bgAlt']}) for c in final_cols])),
        html.Tbody([html.Tr([html.Td(found.iloc[i][c], style={'padding':'10px', 'border':f'1px solid {NVIM["border"]}'}) for c in final_cols]) for i in range(len(found))])
    ], style={"width": "100%", "borderCollapse": "collapse", "fontFamily": "'JetBrains Mono'", "fontSize": "12px"})

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=8050, debug=False)