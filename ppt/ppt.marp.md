---
marp: true
marp: true
theme: gaia
paginate: true
backgroundColor: #ffffff
color: #333333
style: |
  section { font-family: 'Arial', sans-serif; font-size: 26px; }
  h1 { color: #005f99; font-size: 40px; }
  h2 { color: #0077cc; }
  code { background: #f0f0f0; color: #d63384; padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }
  pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
  table { width: 100%; font-size: 22px; border-collapse: collapse; }
  th { background-color: #005f99; color: white; padding: 10px; border: 1px solid #ddd; }
  td { padding: 10px; border: 1px solid #ddd; }
---

<!-- _class: lead -->
# Intelligent DBSCAN Optimizer  
**Pipeline • Objective Function • PSO • Custom DBSCAN**

Disusun oleh:  
**[Nama Anda]**

---

# Agenda Presentasi

1. Simplified Data Flow  
2. Preprocessing Pipeline  
3. Embedding (PCA / UMAP)  
4. PSO Objective Function  
5. Pseudocode Optimizer  
6. Custom DBSCAN  
7. Evaluasi & Visualisasi  

---

# 1. Simplified Data Flow

```

CSV → Preprocessing → Embedding → PSO Optimization → DBSCAN → Evaluation → Visualization

```

Setiap modul memiliki perannya masing-masing untuk menghasilkan cluster optimal berbasis **DBSCAN + PSO**.

---

# 2. Preprocessing Pipeline

File: **preprocessing.py**

### Fungsi Utama
- `load_data()`  
- `select_numeric_features()`  
- `transform()`  
  - **Yeo-Johnson Transform**  
  - **RobustScaler**

### Output
`X_scaled` → matriks numerik terstandarisasi  
**shape:** (n_samples, n_features)

---

# 3. Embedding (Dimensionality Reduction)

File: **embedding.py**

### Metode:
- **PCAEmbedder.fit_transform()**
- **UMAPEmbedder.fit_transform()**

### Alasan:
- Menurunkan dimensi → Euclidean distance lebih stabil  
- Mengurangi noise dimensi tinggi  
- Membuat pola cluster lebih terpisah  

### Output:
`X_embedded` → dimensi 2D/3D

---

# 4. PSO Optimization

File: **optimization.py**

### Struktur Partikel
```

position  → [eps, min_samples]
velocity  → [v_eps, v_min_samples]
pbest     → best position individu
gbest     → best seluruh swarm

```

### Output:
```

best_params = { eps: x, min_samples: y }

```

---

# 5. Objective Function (PSO Evaluator)

### Langkah Penilaian:
1. Decode parameter:  
   `eps`, `min_samples`
2. Jalankan **DBSCAN** pada `X_embedded`
3. Hitung:
   - jumlah cluster  
   - noise ratio  
   - Silhouette Score
4. Terapkan **penalty**:

### Penalti:
- Cluster hanya 1 blob → *hard penalty*  
- Noise > 50% → *heavy penalty*  
- Cluster > 20 → *medium penalty*

### Skor:
Cost =  
```

cost = - SilhouetteScore

````
PSO akan **meminimalkan** cost.

---

# 6. Pseudocode: Intelligent DBSCAN Optimizer

```python
CLASS HyperOptDBSCAN:

    FUNCTION __init__(raw_data, reduction_method='pca'):
        IF reduction_method == 'pca':
            reduced_data = PCA(raw_data)
        ELSE:
            reduced_data = UMAP(raw_data)

    FUNCTION objective_function([eps, min_samples]):
        labels = DBSCAN(eps, round(min_samples))(reduced_data)

        num_clusters = count_unique(labels except -1)
        noise_ratio  = count_noise / total

        IF num_clusters < 2: return 1.0
        IF noise_ratio > 0.5: return 0.5
        IF num_clusters > 20: return 0.5

        silhouette = SilhouetteScore(reduced_data, labels)
        return -silhouette

    FUNCTION run_optimization():
        INIT swarm within bounds

        FOR iteration in range(max_iter):
            FOR particle in swarm:
                cost = objective_function(particle.position)
                update_best_positions()
            update_swarm_velocity_and_position()

        RETURN global_best
````

---

# 7. Custom DBSCAN

File: **clustering.py**

### Implementasi dari nol:

* `fit(X, eps, min_samples)`
* `_region_query()`
* `_expand_cluster()`

### Post-processing:

* `remove_noise()`
* `map_labels()`

Output:
`labels` (cluster ID atau noise = -1)

---

# 8. Evaluation Metrics

File: **evaluation.py**

### Metode:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz
* Cluster Statistics:

  * jumlah cluster
  * jumlah noise
  * noise ratio

Digunakan untuk menilai performa parameter hasil PSO.

---

# 9. Visualization Module

File: **visualization.py**

### Fungsi:

* Plot cluster
* Plot PSO convergence
* Plot metric comparison
* Save PNG + CSV

Contoh:

```
plot_clusters(X_embedded, labels, "DBSCAN Result")
```

---

<!-- _class: lead -->

# Terima Kasih

Ada pertanyaan?

```

