"""
Utils Module - Logger, I/O, Seed Manager
"""

import numpy as np
import pandas as pd
import pickle
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from tqdm import tqdm


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Simple logger with timestamp and levels"""
    
    def __init__(self, name: str = "PSO-DBSCAN", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.logs = []
        self.pbar = None

    # -------------------------
    # PROGRESS BAR SUPPORT
    # -------------------------
    def start_pbar(self, total: int, desc: str = "Processing"):
        """Initialize a tqdm progress bar."""
        self.pbar = tqdm(total=total, desc=desc)
    
    def update_pbar(self, n: int = 1):
        """Update tqdm progress bar."""
        if self.pbar:
            self.pbar.update(n)

    def close_pbar(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    # -------------------------
    # LOGGING FUNCTIONS
    # -------------------------
    def _format_message(self, level: str, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {level}: {message}"

    def info(self, message: str):
        msg = self._format_message("INFO", message)
        self.logs.append(msg)
        if self.verbose:
            print(msg)

    def warning(self, message: str):
        msg = self._format_message("WARN", message)
        self.logs.append(msg)
        if self.verbose:
            print(f"\033[93m{msg}\033[0m")

    def error(self, message: str):
        msg = self._format_message("ERROR", message)
        self.logs.append(msg)
        if self.verbose:
            print(f"\033[91m{msg}\033[0m")

    def success(self, message: str):
        msg = self._format_message("SUCCESS", message)
        self.logs.append(msg)
        if self.verbose:
            print(f"\033[92m{msg}\033[0m")

    def save_logs(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.logs))


# ============================================================================
# SEED MANAGEMENT
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================================
# FILE I/O
# ============================================================================

def ensure_dir(directory: str):
    """Create directory if not exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, filepath: str):
    """Save object as pickle"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle object"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save dict as JSON"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_numpy(array: np.ndarray, filepath: str):
    """Save numpy array"""
    ensure_dir(Path(filepath).parent)
    np.save(filepath, array)


def load_numpy(filepath: str) -> np.ndarray:
    """Load numpy array"""
    return np.load(filepath)


def save_results(
    labels: np.ndarray,
    metrics: Dict,
    output_dir: str,
    prefix: str = "result"
):
    """Save clustering results (labels + metrics)"""
    ensure_dir(output_dir)
    
    # Save labels
    labels_path = f"{output_dir}/{prefix}_labels.npy"
    save_numpy(labels, labels_path)
    
    # Save metrics
    metrics_path = f"{output_dir}/{prefix}_metrics.json"
    save_json(metrics, metrics_path)
    
    # Save as CSV too
    df = pd.DataFrame({
        'sample_id': range(len(labels)),
        'cluster': labels
    })
    csv_path = f"{output_dir}/{prefix}_labels.csv"
    df.to_csv(csv_path, index=False)
    
    return {
        'labels': labels_path,
        'metrics': metrics_path,
        'csv': csv_path
    }


def load_results(output_dir: str, prefix: str = "result") -> Dict:
    """Load clustering results"""
    labels = load_numpy(f"{output_dir}/{prefix}_labels.npy")
    metrics = load_json(f"{output_dir}/{prefix}_metrics.json")
    return {'labels': labels, 'metrics': metrics}

\
# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class Timer:
    """Simple timer for tracking execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = datetime.now()
    
    def stop(self):
        self.end_time = datetime.now()
        return self.elapsed()
    
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()
    
    def format_time(self) -> str:
        seconds = self.elapsed()
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}m"
        else:
            return f"{seconds/3600:.2f}h"
