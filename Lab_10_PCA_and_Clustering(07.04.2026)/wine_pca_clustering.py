import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# =====================================================
# Lab_12_FastAPI_ML_Service(21.04.2026)) Выбор устройства (GPU / MPS / CPU)
# =====================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device:", device)

# =====================================================
# 2) Загрузка данных Wine
# =====================================================
wine = load_wine()

X = wine.data
y = wine.target

wine_data = pd.DataFrame(X, columns=wine.feature_names)
wine_data["target"] = y

print("Shape исходных данных:", wine_data.shape)
print("Первые 5 объектов:")
print(wine_data.head())

print("Информация о датасете:")
print(wine_data.info())

print("Названия классов:")
print(list(wine.target_names))

print("Распределение классов:")
print(wine_data["target"].value_counts().sort_index())

# =====================================================
# 3) Разделение на признаки и целевую переменную
# =====================================================
X = wine_data.drop("target", axis=1)
y = wine_data["target"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =====================================================
# 4) Масштабирование данных
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape X_scaled:", X_scaled.shape)

# =====================================================
# 5) Снижение размерности PCA
# =====================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Shape X_pca:", X_pca.shape)
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print(f"Суммарная объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")

# =====================================================
# 6) Визуализация реальных классов
# =====================================================
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.title("Wine dataset после PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Класс")
plt.show()

# =====================================================
# 7) Эксперименты KMeans
# =====================================================
k_values = [
    2,
    3,
    4,
    5
]

kmeans_results = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10)

    clusters = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)

    kmeans_results.append({
        "K": k,
        "Silhouette": silhouette,
        "ARI": ari,
        "NMI": nmi,
        "Inertia": kmeans.inertia_})

results_kmeans = pd.DataFrame(kmeans_results)
results_kmeans = results_kmeans.sort_values(by="ARI", ascending=False)

print("\n" + "=" * 60)
print("Результаты KMeans")
print("=" * 60)
print(results_kmeans)

# =====================================================
# 8) Обучение лучшей модели KMeans
# =====================================================
best_kmeans_row = results_kmeans.iloc[0]
best_k = int(best_kmeans_row["K"])

best_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10)

best_kmeans_labels = best_kmeans.fit_predict(X_scaled)
best_kmeans_centers_pca = pca.transform(best_kmeans.cluster_centers_)

print("\n" + "=" * 60)
print("Лучшая модель KMeans")
print("=" * 60)
print(f"Лучшее количество кластеров: {best_k}")
print(f"Silhouette: {best_kmeans_row['Silhouette']:.4f}")
print(f"ARI: {best_kmeans_row['ARI']:.4f}")
print(f"NMI: {best_kmeans_row['NMI']:.4f}")

# =====================================================
# 9) Визуализация KMeans
# =====================================================
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_kmeans_labels, cmap="viridis")
plt.scatter(
    best_kmeans_centers_pca[:, 0],
    best_kmeans_centers_pca[:, 1],
    marker="X",
    s=200,
    label="Центроиды")
plt.title("Кластеризация KMeans")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# =====================================================
# 10) Эксперименты DBSCAN
# =====================================================
eps_values = [
    1.5,
    1.7,
    2.0,
    2.2,
    2.5,
    3.0
]

min_samples_values = [
    3,
    5,
    7,
    10
]

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples)

        clusters = dbscan.fit_predict(X_scaled)

        unique_labels = set(clusters)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = (clusters == -1).sum()

        if n_clusters > 1:
            silhouette = silhouette_score(X_scaled, clusters)
        else:
            silhouette = 0

        ari = adjusted_rand_score(y, clusters)
        nmi = normalized_mutual_info_score(y, clusters)

        dbscan_results.append({
            "Eps": eps,
            "Min Samples": min_samples,
            "Clusters": n_clusters,
            "Noise": noise_count,
            "Silhouette": silhouette,
            "ARI": ari,
            "NMI": nmi})

results_dbscan = pd.DataFrame(dbscan_results)
results_dbscan = results_dbscan.sort_values(by="ARI", ascending=False)

print("\n" + "=" * 60)
print("Результаты DBSCAN")
print("=" * 60)
print(results_dbscan)

# =====================================================
# 11) Обучение лучшей модели DBSCAN
# =====================================================
best_dbscan_row = results_dbscan.iloc[0]

best_eps = best_dbscan_row["Eps"]
best_min_samples = int(best_dbscan_row["Min Samples"])

best_dbscan = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples)

best_dbscan_labels = best_dbscan.fit_predict(X_scaled)

print("\n" + "=" * 60)
print("Лучшая модель DBSCAN")
print("=" * 60)
print(f"Лучшее значение eps: {best_eps}")
print(f"Лучшее значение min_samples: {best_min_samples}")
print(f"Количество кластеров: {int(best_dbscan_row['Clusters'])}")
print(f"Количество шума: {int(best_dbscan_row['Noise'])}")
print(f"Silhouette: {best_dbscan_row['Silhouette']:.4f}")
print(f"ARI: {best_dbscan_row['ARI']:.4f}")
print(f"NMI: {best_dbscan_row['NMI']:.4f}")

# =====================================================
# 12) Визуализация DBSCAN
# =====================================================
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_dbscan_labels, cmap="viridis")
plt.title("Кластеризация DBSCAN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Кластер")
plt.show()

# =====================================================
# 13) Итоговое сравнение моделей
# =====================================================
comparison = pd.DataFrame({
    "Модель": ["KMeans", "DBSCAN"],
    "Параметры": [
        f"k={best_k}",
        f"eps={best_eps}, min_samples={best_min_samples}"
    ],
    "Silhouette": [
        best_kmeans_row["Silhouette"],
        best_dbscan_row["Silhouette"]
    ],
    "ARI": [
        best_kmeans_row["ARI"],
        best_dbscan_row["ARI"]
    ],
    "NMI": [
        best_kmeans_row["NMI"],
        best_dbscan_row["NMI"]
    ]
})

comparison = comparison.sort_values(by="ARI", ascending=False)

print("\n" + "=" * 60)
print("Итоговое сравнение моделей")
print("=" * 60)
print(comparison)

# =====================================================
# 14) Визуализация сравнения
# =====================================================
plt.figure(figsize=(8, 5))
plt.bar(comparison["Модель"], comparison["ARI"])
plt.title("Сравнение ARI моделей")
plt.xlabel("Модель")
plt.ylabel("ARI")
plt.show()

# =====================================================
# 15) Финальный вывод
# =====================================================
best_model = comparison.iloc[0]["Модель"]
best_ari = comparison.iloc[0]["ARI"]

print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)
print(f"Лучшая модель: {best_model}")
print(f"Лучший ARI: {best_ari:.4f}")