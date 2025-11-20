import os
# Set OMP_NUM_THREADS to avoid excessive CPU usage warnings
os.environ.setdefault('OMP_NUM_THREADS', '8')

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import json

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

# Define colors for up to 10 clusters to be safe
CLUSTER_COLORS_LIST = [
    NVIM['red'], NVIM['green'], NVIM['yellow'], NVIM['blue'], NVIM['purple'], 
    NVIM['cyan'], NVIM['orange'], '#be5046', '#35b209', '#d19a66'
]

# ====================================================
# 1. Load dataset & Feature Selection
# ====================================================
CLUSTERING_COLS = [
    'TINGGI', 'BERAT', 'NADI', 'PERNAPASAN', 'SUHU', 
    'HB', 'LEUKOSIT', 'LED', 'TROMBOSIT', 
    'BILIRUBIN_TOTAL', 'BILIRUBIN_DIRECT', 'ALKALINE_PHOSPAT',
    'SGPT', 'SGOT', 'GAMMA_GT', 
    'KOLEST_TOTAL', 'TRIGLISERIDA', 'HDL_KOLEST', 'LDL_KOLEST', 
    'UREUM', 'KREATININ', 'ASAM_URAT_GINJAL', 
    'GULA_DARAH_PUASA', 'GULA_DARAH_2JAMPP', 
    'SISTOLIK', 'DIASTOLIK', 'BMI_CALC'
]

PROFILE_COLS = [
    'BMI_CALC', 'GULA_DARAH_PUASA', 'GULA_DARAH_2JAMPP', 
    'KOLEST_TOTAL', 'TRIGLISERIDA', 'LDL_KOLEST',
    'ASAM_URAT_GINJAL', 'UREUM', 'KREATININ', 
    'SGPT', 'SISTOLIK', 'DIASTOLIK'
]

try:
    df = pd.read_csv("Cleaned_Pasien_MCU_LastVisit.csv")
    df.columns = df.columns.str.strip()
    if 'BADGE' in df.columns:
        df['BADGE'] = df['BADGE'].astype(str).str.replace(r'\.0$', '', regex=True)
    else:
        df['BADGE'] = [str(i) for i in range(len(df))]
    for col in CLUSTERING_COLS:
        if col not in df.columns: df[col] = 0
            
except FileNotFoundError:
    print("Dataset not found. Generating Dummy Data...")
    N = 150
    data = {
        'BADGE': [str(1000+i) for i in range(N)],
        'BMI_CALC': np.random.normal(24, 4, N)
    }
    for col in CLUSTERING_COLS: 
        if col not in data: data[col] = np.random.normal(100, 10, N)
    df = pd.DataFrame(data)

# ====================================================
# 2. Preprocessing & Noise Injection (STATIC ONCE)
# ====================================================
df_cluster = df.copy()
if 'BMI_CALC' not in df_cluster.columns or df_cluster['BMI_CALC'].sum() == 0:
    df_cluster['BMI_CALC'] = (df_cluster['BERAT'] / 10) / ((df_cluster['TINGGI'] / 1000) ** 2)

df_filled = df_cluster[CLUSTERING_COLS].fillna(df_cluster[CLUSTERING_COLS].median())
# Jittering to prevent singularity
noise = np.random.normal(0, 0.0001, df_filled.shape)
df_filled = df_filled + noise

# Scale Data (Global)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filled)

# t-SNE (Global - Calculated ONCE to save time)
# We only re-color points when K changes, we don't need to move them.
n_samples = len(df_filled)
perp = min(30, n_samples - 1) if n_samples > 1 else 1
try:
    tsne = TSNE(n_components=3, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    tsne_3d = tsne.fit_transform(scaled_data)
except Exception:
    tsne_3d = np.random.rand(n_samples, 3)

# Base dataframe for visualization
viz_base_df = df.copy()
viz_base_df['tSNE_1'] = tsne_3d[:, 0]
viz_base_df['tSNE_2'] = tsne_3d[:, 1]
viz_base_df['tSNE_3'] = tsne_3d[:, 2]
if 'BMI_CALC' in df_cluster.columns:
    viz_base_df['BMI_CALC'] = df_cluster['BMI_CALC'].round(1)

valid_profile_cols = [c for c in PROFILE_COLS if c in viz_base_df.columns]
feature_options = [{'label': col, 'value': col} for col in valid_profile_cols]

# ====================================================
# 5. DASH APP
# ====================================================
app = dash.Dash(__name__, title="MCU Analysis Final")

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            * {{ outline: none !important; box-sizing: border-box; }}
            body, html {{ background: {NVIM['bg']} !important; margin: 0; padding: 0; color: {NVIM['fg']}; font-family: 'JetBrains Mono', monospace; }}
            .js-plotly-plot, .plot-container {{ background: {NVIM['bg']} !important; }}
            
            /* GRID LAYOUT */
            .viz-container {{
                display: grid;
                grid-template-columns: 2.5fr 1.5fr;
                gap: 20px;
                padding: 20px 20px 0 20px;
                max-width: 1800px;
                margin: 0 auto;
            }}
            .viz-col-right {{ display: flex; flex-direction: column; gap: 20px; }}
            .search-container {{ padding: 20px; max-width: 1800px; margin: 0 auto; }}
            
            /* CARDS */
            .bento-card {{
                background-color: {NVIM['bg']};
                border: 1px solid {NVIM['border']};
                border-radius: 8px;
                display: flex; flex-direction: column;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            .card-header {{
                background-color: {NVIM['bgAlt']};
                padding: 10px 16px;
                border-bottom: 1px solid {NVIM['border']};
                font-size: 13px;
                display: flex; justify-content: space-between; align-items: center;
            }}
            .card-body {{ padding: 16px; flex-grow: 1; position: relative; }}

            /* CONTROLS */
            .search-input {{
                width: 100%; padding: 12px;
                background-color: {NVIM['bgAlt']};
                border: 1px solid {NVIM['border']};
                color: {NVIM['fg']}; outline: none; border-radius: 4px;
                font-family: 'JetBrains Mono'; font-size: 14px;
            }}
            .search-input:focus {{ border-color: {NVIM['blue']}; }}
            
            .table-scroll {{
                margin-top: 20px; overflow-x: auto; max-height: 400px;
                overflow-y: auto; border: 1px solid {NVIM['border']}; border-radius: 4px;
            }}

            /* OVERRIDES */
            .rc-slider-track {{ background-color: {NVIM['blue']} !important; }}
            .rc-slider-handle {{ border-color: {NVIM['blue']} !important; background-color: {NVIM['bgAlt']} !important; }}
            .rc-slider-mark-text {{ color: {NVIM['fg']} !important; font-family: 'JetBrains Mono'; }}
            .Select-control {{ background-color: {NVIM['bgAlt']} !important; border: 1px solid {NVIM['border']} !important; }}
            .Select-menu-outer {{ background-color: {NVIM['bgAlt']} !important; border: 1px solid {NVIM['border']} !important; }}
            .Select-value-label {{ color: {NVIM['fg']} !important; }}

            /* LOADING SPINNER CUSTOM */
            ._dash-loading {{
                color: {NVIM['purple']};
                font-family: 'JetBrains Mono';
            }}
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
    # STORE UNTUK DATA CLUSTERING (Client Side)
    dcc.Store(id='cluster-store'),

    # --- HEADER & CONTROLS ---
    html.Div([
        html.Div([
            html.Div("// MCU_DASHBOARD_FINAL", style={"fontSize": "12px", "color": NVIM["comment"]}),
            html.H1("fn medical_clustering()", style={"color": NVIM["purple"], "fontSize": "24px", "margin": "4px 0"}),
        ]),
        
        # CONTROL K-MEANS SLIDER
        html.Div([
            html.Label([
                html.Span("n_clusters (K): ", style={"color": NVIM['yellow']}),
                html.Span(id="k-value-display", style={"color": NVIM['fg'], "fontWeight": "bold"})
            ], style={"fontSize": "12px", "marginBottom": "5px", "display": "block"}),
            
            dcc.Slider(
                id='k-cluster-slider',
                min=2, max=5, step=1, value=5,
                marks={i: {'label': str(i), 'style': {'color': NVIM['fg']}} for i in range(2, 6)},
            )
        ], style={"width": "300px", "paddingRight": "20px"}),

        html.Div([
            html.Span(id="silhouette-display", style={"color": NVIM['green'], "fontWeight": "bold", "fontSize": "14px"}),
        ])
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "16px 24px", "backgroundColor": NVIM["statusline"]}),

    # --- LOADING WRAPPER ---
    dcc.Loading(
        id="loading-graphs",
        type="cube",
        color=NVIM['cyan'],
        children=html.Div(className="viz-container", children=[
            
            # LEFT: 3D PLOT
            html.Div(className="bento-card", style={"minHeight": "720px"}, children=[
                html.Div(className="card-header", children=[
                    html.Span("t-SNE Projection (3D)", style={"color": NVIM['cyan']}),
                    html.Span("[Interactive]", style={"color": NVIM['comment'], "fontSize": "11px"})
                ]),
                html.Div(className="card-body", children=[
                    dcc.Graph(id="3d-plot", style={"height": "600px"}),
                    html.Div([
                        html.Label([html.Span("simulation_step: ", style={"color": NVIM["purple"]}), html.Span("0", id="slider-output", style={"color": NVIM["orange"]})], style={"marginBottom": "10px", "fontSize": "12px", "display": "block"}),
                        dcc.Slider(id='step-slider', min=0, max=10, step=1, value=10)
                    ], style={"marginTop": "10px"})
                ])
            ]),

            # RIGHT: RADAR & BOX
            html.Div(className="viz-col-right", children=[
                # RADAR
                html.Div(className="bento-card", style={"flex": "1"}, children=[
                    html.Div(className="card-header", children=[
                        html.Span("Cluster Profile (Radar)", style={"color": NVIM['yellow']})
                    ]),
                    html.Div(className="card-body", children=[
                        dcc.Graph(id="radar-plot", style={"height": "300px"}, config={'displayModeBar': False})
                    ])
                ]),
                # BOXPLOT
                html.Div(className="bento-card", style={"flex": "1"}, children=[
                    html.Div(className="card-header", children=[
                        html.Span("Distribution (Box)", style={"color": NVIM['blue']}),
                        dcc.Dropdown(
                            id='feature-dropdown', 
                            options=feature_options, 
                            value=feature_options[0]['value'] if feature_options else None, 
                            clearable=False, 
                            style={'width': '180px', 'fontSize': '12px', 'color': '#000'}
                        )
                    ]),
                    html.Div(className="card-body", children=[
                        dcc.Graph(id="box-plot", style={"height": "300px"}, config={'displayModeBar': False})
                    ])
                ])
            ])
        ])
    ),

    # --- SEARCH SECTION ---
    html.Div(className="search-container", children=[
        html.Div(className="bento-card", children=[
            html.Div(className="card-header", children=[
                html.Span("Search Patient Data", style={"color": NVIM['orange']})
            ]),
            html.Div(className="card-body", children=[
                dcc.Input(
                    id="search-badge", 
                    type="text", 
                    placeholder="Ketik BADGE ID dan Tekan Enter...", 
                    debounce=True, 
                    className="search-input"
                ),
                dcc.Loading(
                    id="loading-search", 
                    type="dot", 
                    color=NVIM['purple'], 
                    children=html.Div(id="search-result", className="table-scroll")
                )
            ])
        ])
    ]),

    html.Div([html.Span("}", style={"color": NVIM["fg"]}), html.Span("// end of analysis", style={"color": NVIM["comment"], "marginLeft": "12px"})], style={"textAlign": "center", "padding": "20px"})

], style={"backgroundColor": NVIM["bg"], "minHeight": "100vh"})

# ====================================================
# CALLBACKS: CLUSTERING LOGIC
# ====================================================

# 1. HEAVY LIFTING: Run K-Means when Slider K changes
@app.callback(
    [Output("cluster-store", "data"), Output("k-value-display", "children"), Output("silhouette-display", "children")],
    [Input("k-cluster-slider", "value")]
)
def run_clustering(k):
    if k is None: k = 5
    
    # Run K-Means
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled_data)
        # Safe cast to standard python list of ints for JSON serialization
        labels_list = [int(x) for x in labels]
    except Exception as e:
        print(f"Clustering Error: {e}")
        labels_list = [0] * len(viz_base_df)
        labels = np.zeros(len(viz_base_df))

    # Calc Metric
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(scaled_data, labels)
    else:
        sil_score = 0.0
    
    silhouette_text = f"Silhouette Score: {sil_score:.3f}"
    
    # Return data to Store
    store_data = {
        'labels': labels_list,
        'k': k
    }
    return store_data, str(k), silhouette_text

# 2. VISUALIZATION: 3D Plot (Depends on Store & Step Slider)
@app.callback(
    Output("3d-plot", "figure"),
    [Input("step-slider", "value"), Input("cluster-store", "data")]
)
def update_3d(step, store_data):
    if step is None: step = 10
    if store_data is None: return go.Figure()
    
    labels = store_data['labels']
    k = store_data['k']
    
    # Create Local DF for plotting
    local_df = viz_base_df.copy()
    local_df['Cluster'] = [str(x) for x in labels]
    
    # 1. Calculate Trajectory (On the fly - it's fast enough)
    centroids = local_df.groupby('Cluster')[['tSNE_1', 'tSNE_2', 'tSNE_3']].mean()
    
    # Color Map Dynamic
    local_color_map = {str(i): CLUSTER_COLORS_LIST[i] for i in range(k)}
    
    custom_hover = []
    for _, row in local_df.iterrows():
        txt = (f"BADGE: {row.get('BADGE', '-')}<br>"
               f"Cluster: {row.get('Cluster', '-')}<br>"
               f"BMI: {row.get('BMI_CALC', '-')}<br>"
               f"Gula: {row.get('GULA_DARAH_PUASA', '-')}<br>"
               f"Tensi: {row.get('SISTOLIK', '-')}/{row.get('DIASTOLIK', '-')}")
        custom_hover.append(txt)
        
    fig = go.Figure()
    safe_colors = [local_color_map.get(c, NVIM['fg']) for c in local_df['Cluster']]
    
    # Trace 1: Patients
    fig.add_trace(go.Scatter3d(
        x=local_df['tSNE_1'], y=local_df['tSNE_2'], z=local_df['tSNE_3'], mode='markers',
        marker=dict(size=4, opacity=0.6, color=safe_colors),
        hovertext=custom_hover, hoverinfo='text', name='Patients'
    ))
    
    # Trace 2: Centroids Animation
    # Origin (mean of all data)
    origin = local_df[['tSNE_1', 'tSNE_2', 'tSNE_3']].mean().values
    
    # Calculate current step position for each centroid
    frac = step / 10.0
    
    for cl in local_df['Cluster'].unique():
        if cl not in centroids.index: continue
        
        target = centroids.loc[cl].values
        # Lerp
        current_pos = origin + (target - origin) * frac
        
        c_color = local_color_map.get(cl, NVIM['fg'])
        
        # Trajectory Line (From Origin to Current)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], current_pos[0]], 
            y=[origin[1], current_pos[1]], 
            z=[origin[2], current_pos[2]], 
            mode='lines', line=dict(color=c_color, width=4), showlegend=False
        ))
        
        # Centroid Marker
        fig.add_trace(go.Scatter3d(
            x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]], mode='markers',
            marker=dict(size=12, symbol='diamond', color=c_color, line=dict(color='#fff', width=1)),
            name=f'Centroid {cl}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor=NVIM['bgAlt'], gridcolor=NVIM['border'], title=dict(text='tSNE-1', font=dict(color=NVIM['cyan']))),
            yaxis=dict(backgroundcolor=NVIM['bgAlt'], gridcolor=NVIM['border'], title=dict(text='tSNE-2', font=dict(color=NVIM['cyan']))),
            zaxis=dict(backgroundcolor=NVIM['bgAlt'], gridcolor=NVIM['border'], title=dict(text='tSNE-3', font=dict(color=NVIM['cyan']))),
        ),
        paper_bgcolor=NVIM['bg'], plot_bgcolor=NVIM['bg'], font=dict(family="JetBrains Mono", color=NVIM['fg']),
        margin=dict(l=0, r=0, t=0, b=0), legend=dict(bgcolor=NVIM['bgAlt'], bordercolor=NVIM['border'])
    )
    return fig

# 3. VISUALIZATION: Radar & Box (Depends on Store)
@app.callback(
    [Output("radar-plot", "figure"), Output("box-plot", "figure")],
    [Input("cluster-store", "data"), Input("feature-dropdown", "value")]
)
def update_charts(store_data, feature):
    if store_data is None: return go.Figure(), go.Figure()
    
    labels = store_data['labels']
    k = store_data['k']
    local_df = viz_base_df.copy()
    local_df['Cluster'] = [str(x) for x in labels]
    local_color_map = {str(i): CLUSTER_COLORS_LIST[i] for i in range(k)}
    
    # --- RADAR ---
    radar_fig = go.Figure()
    if valid_profile_cols:
        # Normalize for radar
        scaler_radar = MinMaxScaler()
        radar_source = local_df[valid_profile_cols].fillna(local_df[valid_profile_cols].median())
        df_norm = pd.DataFrame(scaler_radar.fit_transform(radar_source), columns=valid_profile_cols)
        df_norm['Cluster'] = local_df['Cluster'].values
        radar_data = df_norm.groupby('Cluster').mean().reset_index()
        
        cats = valid_profile_cols
        for cl in radar_data['Cluster'].unique():
            vals = radar_data[radar_data['Cluster'] == cl][cats].values.flatten().tolist()
            vals += vals[:1]
            cats_closed = cats + [cats[0]]
            c_color = local_color_map.get(str(cl), NVIM['fg'])
            radar_fig.add_trace(go.Scatterpolar(r=vals, theta=cats_closed, fill='toself', name=f'Cluster {cl}', line=dict(color=c_color)))
            
    radar_fig.update_layout(
        polar=dict(bgcolor=NVIM['bg'], radialaxis=dict(visible=True, showticklabels=False)),
        paper_bgcolor=NVIM['bg'], font=dict(color=NVIM['fg']),
        margin=dict(l=40, r=40, t=20, b=20), legend=dict(font=dict(size=10))
    )
    
    # --- BOX PLOT ---
    if not feature: feature = 'GULA_DARAH_PUASA'
    box_fig = px.box(local_df, x="Cluster", y=feature, color="Cluster", color_discrete_map=local_color_map)
    box_fig.update_layout(
        paper_bgcolor=NVIM['bg'], plot_bgcolor=NVIM['bg'], font=dict(color=NVIM['fg']),
        margin=dict(l=40, r=40, t=40, b=40), showlegend=False
    )
    
    return radar_fig, box_fig

# 4. VISUALIZATION: Search
@app.callback(Output("search-result", "children"), Input("search-badge", "value"), State("cluster-store", "data"))
def search(badge, store_data):
    if not badge: return html.Div("Waiting input...", style={'color': NVIM['comment'], 'padding': '10px'})
    
    # Reconstruct data with current cluster labels
    local_df = viz_base_df.copy()
    if store_data:
        local_df['Cluster'] = [str(x) for x in store_data['labels']]
    else:
        local_df['Cluster'] = '?'

    found = local_df[local_df['BADGE'].astype(str).str.contains(badge, na=False)]
    if found.empty: return html.Div("Not found", style={'color': NVIM['red'], 'padding': '10px'})
    
    cols = ['BADGE', 'Cluster', 'BMI_CALC', 'GULA_DARAH_PUASA', 'KOLEST_TOTAL', 'SISTOLIK', 'DIASTOLIK']
    final_cols = [c for c in cols if c in found.columns]
    
    return html.Table([
        html.Thead(html.Tr([html.Th(c, style={'padding':'8px', 'border':f'1px solid {NVIM["border"]}'}) for c in final_cols])),
        html.Tbody([html.Tr([html.Td(found.iloc[i][c], style={'padding':'8px', 'border':f'1px solid {NVIM["border"]}'}) for c in final_cols]) for i in range(len(found))])
    ], style={'width':'100%', 'color': NVIM['fg'], 'borderCollapse': 'collapse'})

# 5. Slider Output
@app.callback(Output("slider-output", "children"), Input("step-slider", "value"))
def update_slider_val(val): return str(val)

if __name__ == "__main__":
    print("Running server...")
    app.run(host="127.0.0.1", port=8050, debug=False)