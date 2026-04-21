import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.colors import ListedColormap

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
# 2) Загрузка датасета Breast Cancer
# =====================================================
data = load_breast_cancer()
X_raw = data.data
y = data.target
feature_names = data.feature_names

print("Shape исходных данных:", X_raw.shape)
print("Первые 5 объектов:")
print(X_raw[:5])

print("Первые 5 меток:")
print(y[:5])

# =====================================================
# 3) Масштабирование признаков
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Первые 5 объектов после StandardScaler:")
print(X_scaled[:5])

# =====================================================
# 4) Выбор двух наиболее значимых признаков
# =====================================================
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_scaled, y)

selected_indices = selector.get_support(indices=True)
selected_features = feature_names[selected_indices]

print("Выбранные признаки:")
print(selected_features)

# =====================================================
# 5) Разделение на обучающую и тестовую выборки
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# =====================================================
# 6) Визуализация обучающей выборки
# =====================================================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolor="k",
    alpha=0.7)

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=["Malignant (0)", "Benign (1)"])

plt.title("Breast Cancer Dataset (2 selected features)")
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()

# =====================================================
# 7) Метод Парзеновского окна
# =====================================================
class ParzenWindowClassifier:
    def __init__(self, h=0.5):
        self.h = h
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def _gaussian_kernel(self, distance):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distance ** 2))

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], len(self.classes)))

        for idx, cls in enumerate(self.classes):
            X_cls = self.X_train[self.y_train == cls]

            for i, x_query in enumerate(X):
                dists = np.linalg.norm(X_cls - x_query, axis=1)
                kernel_vals = self._gaussian_kernel(dists / self.h)
                probs[i, idx] = np.sum(kernel_vals)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

# =====================================================
# 8) Метод потенциальных функций
# =====================================================
class PotentialFunctionClassifier:
    def __init__(self, h=1.0, epochs=10):
        self.h = h
        self.epochs = epochs
        self.support_vectors_ = []
        self.charges_ = []

    def _kernel(self, u):
        return np.exp(-1 * (u ** 2))

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.support_vectors_ = []
        self.charges_ = []

        y_signed = np.where(y == 0, -1, 1)

        for epoch in range(self.epochs):
            errors = 0

            for i in range(n_samples):
                x_curr = X[i]
                y_true = y_signed[i]

                decision_val = self._decision_function_single(x_curr)
                y_pred = 1 if decision_val > 0 else -1

                if y_pred != y_true:
                    self.support_vectors_.append(x_curr)
                    self.charges_.append(y_true)
                    errors += 1

            if errors == 0:
                print(f"Сходимость достигнута на эпохе {epoch}")
                break

    def _decision_function_single(self, x):
        if not self.support_vectors_:
            return 0

        support_vectors = np.array(self.support_vectors_)
        charges = np.array(self.charges_)

        dists = np.linalg.norm(support_vectors - x, axis=1)
        weights = self._kernel(dists / self.h)

        return np.sum(charges * weights)

    def predict(self, X):
        preds = []

        for x in X:
            score = self._decision_function_single(x)
            preds.append(1 if score > 0 else 0)

        return np.array(preds)

# =====================================================
# 9) Вспомогательная функция для отрисовки границ
# =====================================================
def plot_decision_boundary(clf, X, y, title=""):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=ListedColormap(["#FF0000", "#0000FF"]),
        edgecolors="k")

    plt.title(title)
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])

# =====================================================
# 10) Подбор гиперпараметра h для Парзеновского окна
# =====================================================
h_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

best_h_parzen = None
best_acc_parzen = 0

print("\n" + "=" * 60)
print("Подбор параметра h для Парзеновского окна")
print("=" * 60)

for h in h_values:
    clf = ParzenWindowClassifier(h=h)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"h = {h}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 60)

    if acc > best_acc_parzen:
        best_acc_parzen = acc
        best_h_parzen = h

print(f"\nЛучшее h для Парзеновского окна: {best_h_parzen}")
print(f"Лучшая accuracy: {best_acc_parzen:.4f}")

# =====================================================
# 11) Подбор гиперпараметра h для метода потенциальных функций
# =====================================================
best_h_potential = None
best_acc_potential = 0

print("\n" + "=" * 60)
print("Подбор параметра h для метода потенциальных функций")
print("=" * 60)

for h in h_values:
    clf = PotentialFunctionClassifier(h=h, epochs=15)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"h = {h}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Количество потенциалов: {len(clf.charges_)}")
    print("-" * 60)

    if acc > best_acc_potential:
        best_acc_potential = acc
        best_h_potential = h

print(f"\nЛучшее h для метода потенциальных функций: {best_h_potential}")
print(f"Лучшая accuracy: {best_acc_potential:.4f}")

# =====================================================
# 12) Обучение лучших моделей
# =====================================================
best_parzen = ParzenWindowClassifier(h=best_h_parzen)
best_parzen.fit(X_train, y_train)

best_potential = PotentialFunctionClassifier(h=best_h_potential, epochs=15)
best_potential.fit(X_train, y_train)

y_pred_parzen = best_parzen.predict(X_test)
y_pred_potential = best_potential.predict(X_test)

acc_parzen = accuracy_score(y_test, y_pred_parzen)
acc_potential = accuracy_score(y_test, y_pred_potential)

print("\n" + "=" * 60)
print("Итоговое сравнение моделей")
print("=" * 60)
print(f"Parzen Window Accuracy: {acc_parzen:.4f}")
print(f"Potential Function Accuracy: {acc_potential:.4f}")

# =====================================================
# 13) Визуализация границ решений
# =====================================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(
    best_parzen,
    X_test,
    y_test,
    title=f"Parzen Window (h={best_h_parzen})")

plt.subplot(1, 2, 2)
plot_decision_boundary(
    best_potential,
    X_test,
    y_test,
    title=f"Potential Functions (h={best_h_potential})")

plt.tight_layout()
plt.show()

# =====================================================
# 14) Финальный вывод
# =====================================================
print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)

if acc_parzen > acc_potential:
    print("Лучшая модель: Parzen Window")
    print(f"Accuracy: {acc_parzen:.4f}")
elif acc_potential > acc_parzen:
    print("Лучшая модель: Potential Function")
    print(f"Accuracy: {acc_potential:.4f}")
else:
    print("Обе модели показали одинаковый результат")
    print(f"Accuracy: {acc_parzen:.4f}")

print(f"Лучшее h для Parzen Window: {best_h_parzen}")
print(f"Лучшее h для Potential Function: {best_h_potential}")