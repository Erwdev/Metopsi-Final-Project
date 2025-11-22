# Metopsi-Final-Project

Repository untuk preprocessing data MCU dan clustering dengan K-Means teroptimasi.

## Anggota

- Tegar Prasetyo (23/520277/PA/22364)
- Benedictus Erwin Widianto (23/520176/PA/22350)

## Overview

Pipeline modular untuk **Medical Check-Up (MCU) Patient Clustering** dengan fitur engineering berbasis domain medis:

```
[Raw Data] → [Feature Selection] → [Elbow Method] → [K-Means] → [PSO Refinement] → [Results]
```

- **Preprocessing:** Load, handle missing, normalize
- **Feature Engineering:** 7 kategori medis (41 fitur total)
- **K Optimization:** Elbow method untuk K optimal
- **WCSS Refinement:** PSO untuk fine-tune centroid
- **Clustering:** K-Means dengan K optimal

---

## Module Architecture

```
src/
├── k_mean.py ..................... Core K-Means algorithm
│   ├── KMeansConfig .............. Configuration object
│   ├── DataPreprocessor .......... Load, normalize, handle missing
│   └── KMeansClusterer ........... Clustering implementation
│
├── optimization.py ............... K optimization & PSO WCSS refinement
│   ├── ElbowOptimizer ............ Find optimal K
│   ├── PSOOptimizer .............. Fine-tune WCSS
│   └── OptimizationPipeline ...... Orchestrate optimization
│
├── pso.py ........................ Particle Swarm Optimization
│   └── PSOWCSSOptimizer .......... Optimize centroids for minimum WCSS
│
├── feature_engineering.py ........ Feature management (centralized)
│   ├── FEATURE_CATEGORIES ........ 7 medical domains
│   ├── FeatureSelector ........... Select features by category
│   └── setup_logger() ............ Shared logger utility
│
├── main.py ....................... End-to-end pipeline orchestration
│   └── ClusteringPipeline ........ Load→Select→Optimize→Cluster→Save
│
└── training.py ................... Training examples & CLI
    ├── train_clustering_model() .. Generic trainer
    └── 5 example functions ....... Different use cases
```

---

## Feature Categories (41 Total Features)

| Kategori | Jumlah | Fitur |
|----------|--------|-------|
| **Hematologi** | 9 | HB, LEUKOSIT, LED, EOSINOPIL, BASOPIL, SEGMENT, LYMPOSIT, MONOSIT, TROMBOSIT |
| **Liver Function** | 7 | BILIRUBIN (3), ALKALINE_PHOSPAT, SGPT, SGOT, GAMMA_GT |
| **Kidney Function** | 3 | UREUM, KREATININ, ASAM_URAT_GINJAL |
| **Lipid Profile** | 4 | KOLEST_TOTAL, TRIGLISERIDA, HDL, LDL |
| **Glucose Metabolism** | 2 | GULA_DARAH_PUASA, GULA_DARAH_2JAMPP |
| **Vital Signs** | 5 | TINGGI, BERAT, NADI, PERNAPASAN, SUHU |
| **Urine Test** | 4 | UROBILINOGEN, BILIRUBIN_1, ASAM_URAT_1, TRIPLE_PHOSP_1 |

---

## Quick Usage

### 1. Basic Clustering (Elbow + K-Means)
```python
from main import ClusteringPipeline

pipeline = ClusteringPipeline()
pipeline.load_data("data/Cleaned_agg_pasien_MCU_2.csv")
pipeline.select_features(categories=['hematologi', 'liver_function'])

# Find optimal K
opt = pipeline.optimize_k(k_range=(2, 10))
k_optimal = opt['optimal_k']

# Cluster
results = pipeline.run_clustering(n_clusters=k_optimal)
pipeline.save_results("output.csv")
```

### 2. With PSO Refinement (Full Pipeline)
```python
# ... same as above until optimize_k ...

# Refine WCSS with PSO
pso_result = pipeline.refine_wcss_with_pso(
    n_clusters=k_optimal,
    n_particles=20,
    n_iterations=50
)

results = pipeline.run_clustering(n_clusters=k_optimal)
```

### 3. Using Training Examples
```bash
cd src
python training.py 1        # Single category (Hematologi)
python training.py 2        # Multiple categories
python training.py 3        # Custom features
python training.py 4        # Fixed K value
python training.py 5        # All categories
python training.py all      # Run all examples
```

---

## Workflow Comparison

### SEBELUM ❌
```
PSO mencari K optimal → Confusing!
```

### SESUDAH ✅
```
1. ELBOW METHOD → Find optimal K
   - Input: data, k_range
   - Output: K optimal
   
2. K-MEANS CLUSTERING → Cluster dengan K optimal
   - Input: data, K
   - Output: labels, WCSS
   
3. PSO REFINEMENT (Optional) → Fine-tune centroids
   - Input: data, K, initial centroids
   - Output: better centroids, lower WCSS
```

**Key Point:** PSO tidak mencari K, PSO mengoptimasi WCSS untuk K tertentu!

---

## Documentation Files

- **ARCHITECTURE_REFACTOR.md** ← Perubahan terbaru & penjelasan PSO role
- **preprocessing_docs.md** ← Data preprocessing steps

---

## Requirements

```
numpy
pandas
scikit-learn
```

Install: `pip install -r requirements.txt`

---

## Dataset

- **Source:** `data/Cleaned_agg_pasien_MCU_2.csv`
- **Rows:** 1,889 patients
- **Columns:** 50+ medical parameters
- **Output:** Cluster labels assigned to each patient

---

## Key Concepts

### Elbow Method
Mencari "elbow point" dalam kurva WCSS vs K untuk menemukan K optimal.
- Hitung WCSS untuk K=2,3,...,10
- Temukan knee point (maximum curvature)
- Return optimal K

### PSO for WCSS Refinement
Mengoptimalkan posisi centroid untuk meminimalkan WCSS dengan K tetap.
- Multiple particles mewakili centroid
- Setiap iterasi: update velocity, position, evaluate WCSS
- Return: centroid terbaik yang minimize WCSS

### K-Means Algorithm
Standard K-Means dengan:
- Random initialization
- Iterative clustering
- Convergence check (centroid shift < tolerance)
- WCSS tracking

---

## Common Tasks

### Select different feature combinations
```python
# Single category
pipeline.select_features(category='hematologi')

# Multiple categories
pipeline.select_features(categories=['hematologi', 'liver_function', 'lipid_profile'])

# Custom features
pipeline.select_features(custom_features=['HB', 'LEUKOSIT', 'SGOT'])
```

### Get cluster statistics
```python
stats = pipeline.get_cluster_statistics()
# Output: cluster sizes & percentages
```

### Get pipeline summary
```python
summary = pipeline.get_summary()
# Output: n_samples, n_features, n_clusters, convergence info
```

---

## File Structure

```
project/
├── data/
│   ├── Cleaned_agg_pasien_MCU_2.csv
│   └── MCU_Pasien_With_Cluster_Labels.csv (output)
│
├── src/
│   ├── k_mean.py
│   ├── optimization.py
│   ├── pso.py
│   ├── feature_engineering.py
│   ├── main.py
│   └── training.py
│
├── README.md (this file)
├── ARCHITECTURE_REFACTOR.md
└── preprocessing_docs.md
```

---

## Next Steps

1. Run example pipeline
2. Explore different feature combinations
3. Compare Elbow K vs PSO-refined results
4. Fine-tune PSO parameters (n_particles, n_iterations)
5. Validate clusters with domain expertise
- `convert_excel.ipynb` — kode preprocessing
- `preprocessing_docs.md` — dokumentasi alur (ringkasan)
- `requirements.txt` — paket yang dibutuhkan
