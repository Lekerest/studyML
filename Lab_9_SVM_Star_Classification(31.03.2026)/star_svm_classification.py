import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# 2) Загрузка данных Star Classification
# =====================================================
star_data = pd.read_csv("star_classification.csv")

print("Shape исходных данных:", star_data.shape)
print("Первые 5 объектов:")
print(star_data.head())

print("Информация о датасете:")
print(star_data.info())

# =====================================================
# 3) Проверка пропущенных значений и дубликатов
# =====================================================
print("Количество пропущенных значений:")
print(star_data.isnull().sum())

print("Количество дубликатов:", star_data.duplicated().sum())

star_data = star_data.drop_duplicates().copy()

# =====================================================
# 4) Подготовка данных
# =====================================================
id_columns = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID"
]

star_data = star_data.drop(id_columns, axis=1)

print("Shape после удаления ID признаков:", star_data.shape)
print("Распределение целевой переменной:")
print(star_data["class"].value_counts())

# =====================================================
# 5) Обработка выбросов
# =====================================================
numeric_features = star_data.select_dtypes(include=["int64", "float64"]).columns

outliers_count = {}

for column in numeric_features:
    q1 = star_data[column].quantile(0.25)
    q3 = star_data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = ((star_data[column] < lower_bound) | (star_data[column] > upper_bound)).sum()
    outliers_count[column] = outliers

    star_data[column] = star_data[column].clip(lower_bound, upper_bound)

outliers_table = pd.DataFrame({
    "Признак": list(outliers_count.keys()),
    "Количество выбросов": list(outliers_count.values())})

print("Количество выбросов по признакам:")
print(outliers_table)

# =====================================================
# 6) Кодирование целевой переменной
# =====================================================
target_encoder = LabelEncoder()
star_data["class_encoded"] = target_encoder.fit_transform(star_data["class"])

print("Классы:")
print(list(target_encoder.classes_))

# =====================================================
# 7) Формирование подвыборки для SVM
# =====================================================
sample_size = 3000

if star_data.shape[0] > sample_size:
    star_sample, _ = train_test_split(
        star_data,
        train_size=sample_size,
        random_state=42,
        stratify=star_data["class_encoded"])
else:
    star_sample = star_data.copy()

print("Shape подвыборки:", star_sample.shape)
print("Распределение классов в подвыборке:")
print(star_sample["class"].value_counts())

# =====================================================
# 8) Разделение на признаки и целевую переменную
# =====================================================
X = star_sample.drop(["class", "class_encoded"], axis=1)
y = star_sample["class_encoded"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =====================================================
# 9) Разделение на обучающую и тестовую выборки
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# =====================================================
# 10) Предобработка данных
# =====================================================
preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

# =====================================================
# 11) Модель SVM из ноутбука
# =====================================================
svm_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(
        kernel="rbf",
        random_state=42))])

# =====================================================
# 12) Сетка параметров gamma и C
# =====================================================
param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": [0.001, 0.01, 0.1, 1]
}

print("Сетка параметров:")
print(param_grid)

# =====================================================
# 13) Подбор параметров GridSearchCV
# =====================================================
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1)

grid_search.fit(X_train, y_train)

# =====================================================
# 14) Предсказание
# =====================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# =====================================================
# 15) Оценка качества
# =====================================================
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Лучшие параметры SVM")
print("=" * 60)
print("Лучшие параметры:", grid_search.best_params_)
print(f"Лучшая accuracy на кросс-валидации: {grid_search.best_score_:.4f}")
print(f"Accuracy на тестовой выборке: {acc:.4f}")

# =====================================================
# 16) Подробный отчет по SVM
# =====================================================
print("\n" + "=" * 60)
print("Отчет по модели SVM")
print("=" * 60)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_))

# =====================================================
# 17) Сравнение параметров
# =====================================================
grid_results = pd.DataFrame(grid_search.cv_results_)

results = pd.DataFrame({
    "C": grid_results["param_classifier__C"],
    "Gamma": grid_results["param_classifier__gamma"],
    "Mean CV Accuracy": grid_results["mean_test_score"]})

results = results.sort_values(by="Mean CV Accuracy", ascending=False)

print("\n" + "=" * 60)
print("Итоговое сравнение параметров")
print("=" * 60)
print(results)

# =====================================================
# 18) Визуализация тепловой карты
# =====================================================
heatmap_data = grid_results.pivot_table(
    index="param_classifier__C",
    columns="param_classifier__gamma",
    values="mean_test_score")

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data.values, aspect="auto")
plt.colorbar(label="Accuracy")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.xlabel("Gamma")
plt.ylabel("C")
plt.title("Тепловая карта accuracy для SVM")

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(
            j,
            i,
            f"{heatmap_data.iloc[i, j]:.3f}",
            ha="center",
            va="center")

plt.show()

# =====================================================
# 19) Финальный вывод
# =====================================================
best_c = grid_search.best_params_["classifier__C"]
best_gamma = grid_search.best_params_["classifier__gamma"]

print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)
print(f"Лучшее значение C: {best_c}")
print(f"Лучшее значение gamma: {best_gamma}")
print(f"Лучшая accuracy: {acc:.4f}")