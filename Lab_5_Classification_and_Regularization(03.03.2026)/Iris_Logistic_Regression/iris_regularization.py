import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

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
# 2) Загрузка данных Iris_Logistic_Regression
# =====================================================
iris = datasets.load_iris()
X = iris.data
y = iris.target

print("Shape исходных данных:", X.shape)
print("Первые 5 объектов:")
print(X[:5])
print("Первые 5 меток:")
print(y[:5])

# =====================================================
# 3) Разделение на обучающую и тестовую выборки
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# =====================================================
# 4) Масштабирование признаков
# =====================================================
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print("Первые 3 строки X_train_std:")
print(X_train_std[:3])

# =====================================================
# 5) L1-регуляризация (Lasso)
# =====================================================
base_l1 = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    max_iter=1000
)
lr_l1 = OneVsRestClassifier(base_l1)
lr_l1.fit(X_train_std, y_train)

# =====================================================
# 6) L2-регуляризация (Ridge)
# =====================================================
lr_l2 = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000
)
lr_l2.fit(X_train_std, y_train)

# =====================================================
# 7) Предсказание и оценка
# =====================================================
y_pred_l1 = lr_l1.predict(X_test_std)
y_pred_l2 = lr_l2.predict(X_test_std)

print(f"Accuracy (L1): {accuracy_score(y_test, y_pred_l1):.2f}")
print(f"Accuracy (L2): {accuracy_score(y_test, y_pred_l2):.2f}")

# =====================================================
# 8) Сравнение весов
# =====================================================
l1_coef_matrix = np.vstack([est.coef_.ravel() for est in lr_l1.estimators_])

print("\nВеса признаков (первый класс):")
print(f"L1 weights: {l1_coef_matrix[0]}")
print(f"L2 weights: {lr_l2.coef_[0]}")

# =====================================================
# 9) GridSearchCV
# =====================================================
param_grid = [
    {
        'estimator__penalty': ['l1'],
        'estimator__solver': ['liblinear'],
        'estimator__C': [0.01, 0.1, 1, 10, 100],
        'estimator__max_iter': [1000]
    }
]

grid_search_l1 = GridSearchCV(
    OneVsRestClassifier(LogisticRegression()),
    param_grid,
    cv=5
)
grid_search_l1.fit(X_train_std, y_train)

print(f"\nЛучшие параметры для L1: {grid_search_l1.best_params_}")
print(f"Лучшая точность для L1: {grid_search_l1.best_score_:.2f}")

param_grid_l2 = [
    {
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000]
    }
]

grid_search_l2 = GridSearchCV(
    LogisticRegression(),
    param_grid_l2,
    cv=5
)
grid_search_l2.fit(X_train_std, y_train)

print(f"Лучшие параметры для L2: {grid_search_l2.best_params_}")
print(f"Лучшая точность для L2: {grid_search_l2.best_score_:.2f}")

# =====================================================
# 10) Визуализация границы решения на двух признаках
# =====================================================
X_vis = iris.data[:, :2]
y_vis = iris.target

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vis, y_vis, test_size=0.3, random_state=42
)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train_v, y_train_v)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.brg)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', cmap=plt.cm.brg)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Границы решения логистической регрессии (Iris_Logistic_Regression)')
plt.show()

# =====================================================
# 11) Задание 1
# =====================================================
Cs = [0.001, 0.01, 0.1, 1, 10]

print("\n" + "=" * 60)
print("Задание 1. Исследование влияния параметра C при L1-регуляризации")
print("=" * 60)

for c in Cs:
    base_model_l1 = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=c,
        max_iter=1000,
        random_state=42
    )

    model_l1 = OneVsRestClassifier(base_model_l1)
    model_l1.fit(X_train_std, y_train)

    y_pred_task_1 = model_l1.predict(X_test_std)
    acc_task_1 = accuracy_score(y_test, y_pred_task_1)

    coef_matrix = np.vstack([est.coef_.ravel() for est in model_l1.estimators_])

    zero_weights = np.sum(coef_matrix == 0)
    total_weights = coef_matrix.size

    print(f"\nC = {c}")
    print(f"Количество нулевых весов: {zero_weights} из {total_weights}")
    print(f"Accuracy: {acc_task_1:.4f}")
    print("Коэффициенты модели:")
    print(coef_matrix)
    print("-" * 60)

# =====================================================
# 12) Задание 2
# =====================================================
print("\n" + "=" * 60)
print("Задание 2. Сравнение L1 и L2 при очень маленьком C = 0.01")
print("=" * 60)

c_small = 0.01

base_l1_small = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=c_small,
    max_iter=1000,
    random_state=42
)
model_l1_small = OneVsRestClassifier(base_l1_small)
model_l1_small.fit(X_train_std, y_train)

model_l2_small = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    C=c_small,
    max_iter=1000,
    random_state=42
)
model_l2_small.fit(X_train_std, y_train)

y_pred_l1_small = model_l1_small.predict(X_test_std)
y_pred_l2_small = model_l2_small.predict(X_test_std)

acc_l1_small = accuracy_score(y_test, y_pred_l1_small)
acc_l2_small = accuracy_score(y_test, y_pred_l2_small)

l1_small_coef_matrix = np.vstack([est.coef_.ravel() for est in model_l1_small.estimators_])
l2_small_coef_matrix = model_l2_small.coef_

print(f"C = {c_small}")
print("\nВесы модели L1:")
print(l1_small_coef_matrix)

print("\nВесы модели L2:")
print(l2_small_coef_matrix)

print(f"\nAccuracy (L1, C={c_small}): {acc_l1_small:.4f}")
print(f"Accuracy (L2, C={c_small}): {acc_l2_small:.4f}")

print("\nВывод:")
print("L2-регуляризация распределяет веса более равномерно между признаками.")
print("L1-регуляризация создает более резкий контраст между признаками,")
print("потому что часть коэффициентов зануляется, а часть остается ненулевой.")