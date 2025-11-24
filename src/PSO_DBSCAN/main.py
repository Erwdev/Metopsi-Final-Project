import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from pso import PSO
from dbscan import DBSCAN

def load_and_preprocess_data(file_path):
    """
    Memuat dan memproses data dari file CSV
    """
    data = pd.read_csv(r"C:\Project\Metopsi-Final-Project\data\Cleaned_Pasien_MCU_LastVisit.csv")
    
    # Jika data memiliki kolom non-numerik, kita bisa drop atau encode
    #data_numeric = data.select_dtypes(include=[np.number])
    selected_features = [
        # Vital signs & basic measurements
        #'TINGGI', 'BERAT', 'NADI', 'SUHU', 'HB',
        
        # Blood chemistry - liver function
        'BILIRUBIN_TOTAL', 'SGPT', 'SGOT', 'ALKALINE_PHOSPAT', 'GAMMA_GT',#no3
        
        # Blood chemistry - kidney function
        #'UREUM', 'KREATININ', 'ASAM_URAT_GINJAL',
        
        # Lipid profile
        #'KOLEST_TOTAL', 'TRIGLISERIDA', 'HDL_KOLEST', 'LDL_KOLEST', #no2
        
        # Blood sugar
        #'GULA_DARAH_PUASA', 'GULA_DARAH_2JAMPP',#no1
        
        # Blood cells
        #'LEUKOSIT', 'LED', 'TROMBOSIT',
        
        # Differential blood count
        #'EOSINOPIL', 'BASOPIL', 'SEGMENT', 'LYMPOSIT', 'MONOSIT'
    ]
    available_features = [f for f in selected_features if f in data.columns]
    data_selected = data[available_features].copy()
    # Standardisasi data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_selected)
    
    return data_scaled, scaler

def fast_fitness(params, data, distances_cache=None):
    """
    Fitness function yang JAUH lebih cepat
    - Mengganti silhouette_score dengan davies-bouldin index
    - Davies-Bouldin jauh lebih cepat (O(n) vs O(n²))
    """
    eps, min_samples = params[0], int(np.clip(params[1], 2, 20))
    
    try:
        dbscan = DBSCAN(epsilon=eps, min_pts=min_samples)
        labels = dbscan.fit(data)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if 0 in unique_labels else 0)
        
        # Minimal clusters needed
        if n_clusters <= 1:
            return -1000
        
        # Get non-noise points
        mask = np.array(labels) != 0
        if np.sum(mask) < 2:
            return -1000
        
        labels_filtered = np.array(labels)[mask]
        data_filtered = data[mask]
        
        # Davies-Bouldin Index (lebih cepat dari silhouette)
        score = davies_bouldin_index(data_filtered, labels_filtered)
        
        # Penalty untuk noise
        noise_ratio = np.sum(np.array(labels) == 0) / len(labels)
        score = score - (noise_ratio * 0.5)
        
        return score
    except Exception as e:
        return -1000


def davies_bouldin_index(data, labels):
    """
    Davies-Bouldin Index (JAUH lebih cepat dari silhouette)
    Lower is better, jadi kita return negative
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return -1000
    
    # Hitung centroid setiap cluster
    centroids = np.array([data[labels == label].mean(axis=0) for label in unique_labels])
    
    # Hitung average distance to centroid
    avg_distances = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_points = data[labels == label]
        avg_distances[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
    
    # Hitung Davies-Bouldin
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                ratio = (avg_distances[i] + avg_distances[j]) / (np.linalg.norm(centroids[i] - centroids[j]) + 1e-10)
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    db_index = db_index / n_clusters
    return -db_index  # Negative karena kita maximize

def main():
    """Main function"""
    try:
        data, scaler = load_and_preprocess_data(r'C:\Project\Metopsi-Final-Project\data\Cleaned_Pasien_MCU_LastVisit.csv')
        print(f"Data shape: {data.shape}")
    except FileNotFoundError:
        print("File tidak ditemukan, menggunakan data dummy...")
        data, _ = make_blobs(n_samples=300, centers=3, random_state=42)
        data = StandardScaler().fit_transform(data)
    
    print("Optimasi DBSCAN dengan PSO (Fast Mode)...")
    
    def objective_func(x):
        return fast_fitness(x, data)
    
    # Reduce iterations untuk testing cepat
    pso = PSO(n_particles=10, n_dims=2, max_iter=15, bounds=(0.1, 5.0))
    best_params, best_score = pso.optimize(objective_func)
    
    print(f"\n=== HASIL OPTIMASI ===")
    print(f"Best eps: {best_params[0]:.4f}")
    print(f"Best min_samples: {int(best_params[1])}")
    print(f"Best score: {best_score:.4f}")
    
    # Apply DBSCAN with best parameters
    dbscan = DBSCAN(epsilon=best_params[0], min_pts=int(best_params[1]))
    labels = dbscan.fit(data)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if 0 in unique_labels else 0)
    n_noise = np.sum(np.array(labels) == 0)
    
    print(f"\n=== HASIL CLUSTERING ===")
    print(f"Jumlah cluster: {n_clusters}")
    print(f"Jumlah noise: {n_noise}")
    print(f"Persentase noise: {(n_noise/len(labels))*100:.1f}%")


if __name__ == "__main__":
    main()