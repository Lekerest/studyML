import warnings
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# =====================================================
# 1) Выбор устройства (GPU / MPS / CPU)
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

print("Shape исходных данных:", X.shape)
print("Первые 5 объектов:")
print(X[:5])

print("Первые 5 меток:")
print(y[:5])

# =====================================================
# 3) Масштабирование признаков
# =====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Первые 5 объектов после StandardScaler:")
print(X[:5])

# =====================================================
# 4) Реализация KNN
# =====================================================
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

# =====================================================
# 5) Функция ручной кросс-валидации
# =====================================================
def cross_validate_knn(X, y, k_neighbors=3, n_folds=5):
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    fold_sizes = np.full(n_folds, len(X) // n_folds, dtype=int)
    fold_sizes[:len(X) % n_folds] += 1

    current = 0
    scores = []

    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size

        X_val = X_shuffled[start:stop]
        y_val = y_shuffled[start:stop]

        X_train = np.concatenate((X_shuffled[:start], X_shuffled[stop:]), axis=0)
        y_train = np.concatenate((y_shuffled[:start], y_shuffled[stop:]), axis=0)

        model = KNN(k=k_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

        current = stop

    return np.mean(scores), scores

# =====================================================
# 6) Подбор оптимального k от 1 до 15
# =====================================================
k_values = list(range(1, 16))
mean_scores = []

print("\n" + "=" * 60)
print("Подбор оптимального k с помощью 5-Fold Cross Validation")
print("=" * 60)

for k in k_values:
    mean_score, fold_scores = cross_validate_knn(X, y, k_neighbors=k, n_folds=5)
    mean_scores.append(mean_score)

    print(f"k = {k}")
    print(f"Scores по фолдам: {np.round(fold_scores, 4)}")
    print(f"Средняя accuracy: {mean_score:.4f}")
    print("-" * 60)

best_k = k_values[np.argmax(mean_scores)]
best_cv_score = max(mean_scores)

print("\nЛучшее значение k:", best_k)
print(f"Лучшая средняя accuracy по CV: {best_cv_score:.4f}")

# =====================================================
# 7) График зависимости accuracy от k
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(k_values, mean_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Зависимость accuracy от числа соседей k")
plt.grid(True)
plt.show()

# =====================================================
# 8) Финальная проверка на train/test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

final_model = KNN(k=best_k)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
final_acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Финальная оценка модели на тестовой выборке")
print("=" * 60)
print(f"Best k: {best_k}")
print(f"Test accuracy: {final_acc:.4f}")

# =====================================================
# 9) Вывод
# =====================================================
print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)
print(f"Оптимальное число соседей: {best_k}")
print(f"Средняя accuracy по кросс-валидации: {best_cv_score:.4f}")
print(f"Accuracy на тестовой выборке: {final_acc:.4f}")