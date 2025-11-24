import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from pathlib import Path

# ============================================================================
# THEME CONFIGURATION (Neovim Style)
# ============================================================================

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