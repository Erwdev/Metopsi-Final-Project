import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
# ============================================================================
# THEME CONFIGURATION (Neovim Style)
# ============================================================================
import math
def set_neovim_style():
    """Applies a dark, Neovim-inspired style to Matplotlib globally."""
    plt.style.use('dark_background')
    mpl.rcParams.update({
        'figure.facecolor': '#1e1e1e',  # Dark Grey/Black
        'axes.facecolor': '#1e1e1e',
        'savefig.facecolor': '#1e1e1e',
        'axes.edgecolor': '#a9b1d6',    # FG color
        'text.color': '#c0caf5',
        'axes.labelcolor': '#c0caf5',
        'xtick.color': '#c0caf5',
        'ytick.color': '#c0caf5',
        'grid.color': '#414868',
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })
NVIM = {
    'bg': '#1e1e1e',
    'fg': '#c0caf5',
    'cyan': '#7aa2f7',
    'pink': '#f7768e',
    'green': '#9ece6a',
    'orange': '#e0af68'
}
# ============================================================================
# PSO CONVERGENCE
# ============================================================================

def plot_pso_convergence(
    history: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "PSO Convergence"
):
    """Plot objective function value over PSO iterations."""
    set_neovim_style() # Ensure style is active
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(1, len(history) + 1)
    # Using a vibrant blue/cyan for the line
    ax.plot(iterations, history, linewidth=2, marker='o', markersize=4, color='#7aa2f7')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Objective Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    # Highlight best point
    if len(history) > 0:
        best_idx = np.argmin(history)
        best_value = history[best_idx]
        ax.scatter(best_idx + 1, best_value, color='#f7768e', s=100, zorder=5, 
                   label=f'Best: {best_value:.4f}', edgecolors='white')
        ax.legend(fontsize=10, facecolor='#292e42', edgecolor='#414868')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()

# ============================================================================
# 3D CLUSTER VISUALIZATION (Interactive / Plotly)
# ============================================================================

def plot_clusters_3d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "3D Cluster Visualization",
    save_path: Optional[str] = None,
    show_noise: bool = True,
    opacity: float = 0.8,
    marker_size: int = 4
):
    """
    Interactive 3D Scatter Plot using Plotly with Dark Theme.
    """
    if X.shape[1] < 3:
        print(f"⚠ Warning: Data has {X.shape[1]} dimensions, using Z=0.")
        X_plot = np.zeros((X.shape[0], 3))
        X_plot[:, :2] = X[:, :2]
    else:
        X_plot = X[:, :3]

    df = pd.DataFrame(X_plot, columns=['x', 'y', 'z'])
    df['Cluster'] = labels.astype(str)
    df['Cluster'] = df['Cluster'].replace('-1', 'Noise')
    
    if not show_noise:
        df = df[df['Cluster'] != 'Noise']
    
    df = df.sort_values('Cluster')

    # Neovim-ish Colors for Clusters
    color_map = {'Noise': 'rgba(100, 100, 100, 0.3)'} # Ghostly grey for noise

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', color='Cluster',
        title=title, opacity=opacity,
        color_discrete_map=color_map,
        template="plotly_dark", # Built-in dark theme
        hover_data={'Cluster': True, 'x': False, 'y': False, 'z': False}
    )

    fig.update_traces(marker=dict(size=marker_size, line=dict(width=0)))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Clusters',
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3',
            bgcolor='#1e1e1e' # Match Matplotlib bg
        )
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ Saved interactive 3D plot: {save_path}")

    fig.show()

# ============================================================================
# FULL REPORT (Matplotlib - Static)
# ============================================================================

def plot_full_report(
    X_embedded: np.ndarray,
    labels: np.ndarray,
    pso_history: List[float],
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """Generate complete visualization report (Static)."""
    set_neovim_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # --- 1. Cluster plot (top-left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    # FIX: Modern Colormap Handling
    if n_clusters > 0:
        # Use Matplotlib's colormaps registry
        cmap_name = 'tab20' if n_clusters <= 20 else 'turbo'
        cmap = mpl.colormaps[cmap_name]
        colors = [cmap(i/n_clusters) for i in range(n_clusters)]
    else:
        colors = []

    # Plot Noise
    if -1 in unique_labels:
        mask = labels == -1
        ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                    c='#565f89', marker='x', s=15, alpha=0.4, label='Noise')

    # Plot Clusters
    cluster_idx = 0
    for label in unique_labels:
        if label == -1: continue
        mask = labels == label
        # Cyclical color usage if n_clusters > colormap length
        color = colors[cluster_idx % len(colors)]
        ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                    color=color, s=25, alpha=0.8, label=f'C{label}')
        cluster_idx += 1
    
    ax1.set_title(f'2D Projection ({n_clusters} clusters)', fontweight='bold')
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    
    # --- 2. PSO convergence (top-right) ---
    ax2 = fig.add_subplot(gs[0, 1])
    iterations = range(1, len(pso_history) + 1)
    ax2.plot(iterations, pso_history, linewidth=2, marker='o', markersize=3, color='#bb9af7')
    
    if len(pso_history) > 0:
        best_idx = np.argmin(pso_history)
        ax2.scatter(best_idx + 1, pso_history[best_idx], color='#f7768e', s=100, zorder=5)
        ax2.text(best_idx + 1, pso_history[best_idx], f" {pso_history[best_idx]:.4f}", 
                 verticalalignment='bottom', color='#f7768e')

    ax2.set_title('PSO Convergence', fontweight='bold')
    
    # --- 3. Metrics bar (bottom-left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    clean_metrics = {k: v for k, v in metrics.items() if v is not None}
    metric_names = list(clean_metrics.keys())
    metric_values = list(clean_metrics.values())
    
    if metric_names:
        # Neovim Palette Bars
        bar_colors = ['#7aa2f7', '#bb9af7', '#e0af68', '#9ece6a']
        bars = ax3.bar(metric_names, metric_values, 
                       color=bar_colors[:len(metric_names)], 
                       alpha=0.9, edgecolor='#1e1e1e')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', 
                     fontsize=9, fontweight='bold', color='white')
    
    ax3.set_title('Validation Metrics', fontweight='bold')
    
    # --- 4. Summary text (bottom-right) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    best_fitness = min(pso_history) if pso_history else 0.0
    
    summary_text = f"""
    CLUSTERING SUMMARY
    {'='*25}
    
    Total Samples  : {len(labels)}
    Clusters Found : {n_clusters}
    Noise Points   : {np.sum(labels == -1)}
    Noise Ratio    : {noise_ratio:.2%}
    
    OPTIMIZATION
    {'='*25}
    Iterations     : {len(pso_history)}
    Best Fitness   : {best_fitness:.4f}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', color='#c0caf5',
             bbox=dict(boxstyle='round', facecolor='#292e42', alpha=0.5, edgecolor='#7aa2f7'))
    
    plt.suptitle('PSO-DBSCAN Final Report', fontsize=16, fontweight='bold', y=0.98, color='white')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
        print(f"✓ Saved report: {save_path}")
    
    plt.show()
    
    
    # [Append this to src/visualization.py]

from sklearn.preprocessing import MinMaxScaler

# Neovim Palette for Plotly
NVIM = {
    'bg': '#1e1e1e',
    'fg': '#c0caf5',
    'cyan': '#7aa2f7',
    'pink': '#f7768e',
    'green': '#9ece6a',
    'orange': '#e0af68'
}


import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
from typing import Optional

# Opsional: Set default renderer agar muncul di browser/notebook/vscode
# pio.renderers.default = "browser" 

def plot_cluster_profiles(
    df: pd.DataFrame, 
    labels: list, 
    output_path: str = "cluster_profiles.html", 
    show_plot: bool = True,
    logger = None
):
    """
    Versi Robust: Menangani banyak fitur dengan dynamic spacing & height.
    """
    try:
        # 1. Persiapan Data
        plot_df = df.copy()
        plot_df['Cluster'] = [str(l) for l in labels]
        
        # Urutkan cluster
        if 'Cluster' in plot_df.columns:
            plot_df.sort_values('Cluster', inplace=True)

        # Melt data
        df_melted = plot_df.melt(id_vars='Cluster', var_name='Feature', value_name='Value')
        
        # --- LOGIC ROBUST UNTUK LAYOUT ---
        n_features = df_melted['Feature'].nunique()
        cols_per_row = 3
        
        # Hitung jumlah baris yang akan terbentuk
        n_rows = math.ceil(n_features / cols_per_row)
        
        # 1. Dynamic Height: 300 pixel per baris (minimal 600px total)
        dynamic_height = max(600, n_rows * 350)
        
        # 2. Dynamic Spacing:
        # Rumus batas plotly: spacing < 1 / (rows - 1)
        # Kita pasang spacing yang aman (misal 50% dari batas maksimalnya)
        if n_rows > 1:
            max_spacing = 1 / (n_rows - 1)
            # Pakai nilai terkecil antara 0.08 (standar) atau batas aman hitungan tadi
            safe_spacing = min(0.08, max_spacing * 0.5) 
        else:
            safe_spacing = 0.08 # Default jika cuma 1 baris

        # 3. Membuat Plot
        fig = px.box(
            df_melted, 
            x='Cluster', 
            y='Value', 
            color='Cluster',
            facet_col='Feature', 
            facet_col_wrap=cols_per_row,
            facet_row_spacing=safe_spacing, # <--- PENTING: Mencegah Error ValueError spacing
            title=f"Cluster Profiles ({n_features} Features)",
            template="plotly_dark",
            height=dynamic_height # <--- PENTING: Agar gambar tidak gepeng
        )
        
        # 4. Finishing Touches
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=80),
            showlegend=False 
        )
        
        # Bebaskan sumbu Y agar tiap fitur punya skala sendiri
        fig.update_yaxes(matches=None, showticklabels=True)
        # Rapikan label X agar tidak berulang-ulang
        fig.update_xaxes(showticklabels=True)

        # 5. Simpan & Tampilkan
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(str(output))
        
        msg = f"Cluster profiles saved to: {output} (Features={n_features}, Rows={n_rows})"
        if logger:
            logger.success(msg)
        else:
            print(f"[INFO] {msg}")

        if show_plot:
            fig.show()

    except Exception as e:
        # Error handling yang lebih informatif tanpa crash total program jika hanya plotting yang gagal
        err_msg = f"Failed to plot cluster profiles: {str(e)}"
        if logger:
            logger.error(err_msg)
        else:
            print(f"[ERROR] {err_msg}")
        # Kita raise lagi jika ingin program berhenti, 
        # atau bisa di-pass jika ingin program lanjut meski plot gagal.
        raise e
   
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .clustering import DBSCAN  # Pastikan import DBSCAN anda benar
from .optimization import _objective_wrapper # Import helper wrapper dari optimization.py

def plot_objective_landscape(
    X: np.ndarray, 
    bounds: dict, 
    objective_kwargs: dict, 
    resolution: int = 20, # Grid 20x20 (Total 400x run DBSCAN, hati-hati jika data besar)
    logger = None
):
    """
    Memvisualisasikan permukaan Objective Function dalam 3D untuk melihat
    apakah landscape-nya 'datar', 'berbukit', atau 'curam'.
    """
    if logger:
        logger.info(f"Generating Objective Landscape ({resolution}x{resolution} grid)...")
        logger.info("This might take a while depending on dataset size...")

    # 1. Buat Grid Parameter
    eps_vals = np.linspace(bounds['eps'][0], bounds['eps'][1], resolution)
    min_samples_vals = np.linspace(bounds['min_samples'][0], bounds['min_samples'][1], resolution)
    
    # Meshgrid untuk plotting
    Eps_grid, MinSamples_grid = np.meshgrid(eps_vals, min_samples_vals)
    Fitness_grid = np.zeros_like(Eps_grid)

    dbscan = DBSCAN()
    
    # 2. Hitung Fitness untuk setiap titik di Grid
    # (Looping manual karena DBSCAN tidak bisa vectorization terhadap parameter)
    total_steps = resolution * resolution
    step = 0
    
    for i in range(resolution):
        for j in range(resolution):
            eps = Eps_grid[i, j]
            # min_samples harus integer
            ms = int(round(MinSamples_grid[i, j])) 
            
            # Run DBSCAN
            labels = dbscan.fit(X, eps=eps, min_samples=ms)
            
            # Hitung Fitness
            # Ingat: Kita ingin MINIMIZE fitness, jadi semakin kecil (negatif) semakin bagus.
            # Namun visualisasi 3D biasanya "Puncak = Bagus". 
            # Jadi nanti kita bisa visualisasikan -fitness atau tetap fitness asli (lembah = bagus).
            fit_val = _objective_wrapper(
                X=X, 
                labels=labels, 
                alpha=objective_kwargs.get('alpha', 0.2), 
                beta=objective_kwargs.get('beta', 0.2), 
                K_min=objective_kwargs.get('K_min', 3), 
                K_max=objective_kwargs.get('K_max', 8)
            )
            
            Fitness_grid[i, j] = fit_val
            
            step += 1
            if step % 50 == 0 and logger:
                # Simple log progress agar tidak dikira hang
                print(f"Landscape scan: {step}/{total_steps} points calculated...")

    # 3. Buat Plot 3D Interaktif
    fig = go.Figure(data=[go.Surface(
        z=Fitness_grid, 
        x=Eps_grid, 
        y=MinSamples_grid,
        colorscale='Viridis_r', # Reverse colorscale (Lembah/Biru = Fitness Rendah/Bagus)
        colorbar=dict(title='Fitness (Lower is Better)')
    )])

    fig.update_layout(
        title='DBSCAN Optimization Landscape',
        scene=dict(
            xaxis_title='Epsilon (eps)',
            yaxis_title='Min Samples',
            zaxis_title='Objective Value (Fitness)'
        ),
        autosize=True,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Tambahkan penjelasan cara baca
    fig.add_annotation(
        text="Lembah (Warna Biru/Ungu) = Solusi Terbaik.<br>Dataran Tinggi (Kuning) = Solusi Buruk.",
        xref="paper", yref="paper",
        x=0, y=1, showarrow=False
    )

    return fig