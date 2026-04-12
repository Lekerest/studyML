import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# 2) Загрузка данных Rain in Australia
# =====================================================
au_rain = pd.read_csv("weatherAUS.csv")

print("Shape исходных данных:", au_rain.shape)
print("Первые 5 объектов:")
print(au_rain.head())

print("Информация о датасете:")
print(au_rain.info())

# =====================================================
# 3) Подготовка данных
# =====================================================
au_rain = au_rain.dropna(subset=["RainTomorrow"]).copy()
au_rain["RainTomorrow"] = au_rain["RainTomorrow"].map({"No": 0, "Yes": 1})

au_rain["Date"] = pd.to_datetime(au_rain["Date"], errors="coerce")
au_rain["Year"] = au_rain["Date"].dt.year
au_rain["Month"] = au_rain["Date"].dt.month
au_rain["Day"] = au_rain["Date"].dt.day
au_rain = au_rain.drop("Date", axis=1)

print("Распределение целевой переменной:")
print(au_rain["RainTomorrow"].value_counts())

# =====================================================
# 4) Разделение на признаки и целевую переменную
# =====================================================
X = au_rain.drop("RainTomorrow", axis=1)
y = au_rain["RainTomorrow"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =====================================================
# 5) Разделение на обучающую и тестовую выборки
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# =====================================================
# 6) Определение числовых и категориальных признаков
# =====================================================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("Числовые признаки:")
print(list(numeric_features))
print("Категориальные признаки:")
print(list(categorical_features))

# =====================================================
# 7) Предобработка данных
# =====================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)])

# =====================================================
# 8) Модели из ноутбука
# =====================================================
model1 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        random_state=42))])

model2 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        random_state=42))])

model3 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1))])

# =====================================================
# 9) Обучение моделей
# =====================================================
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# =====================================================
# 10) Предсказание
# =====================================================
y_pred_1 = model1.predict(X_test)
y_pred_2 = model2.predict(X_test)
y_pred_3 = model3.predict(X_test)

# =====================================================
# 11) Оценка качества
# =====================================================
acc_1 = accuracy_score(y_test, y_pred_1)
acc_2 = accuracy_score(y_test, y_pred_2)
acc_3 = accuracy_score(y_test, y_pred_3)

print("\n" + "=" * 60)
print("Точность моделей")
print("=" * 60)
print(f"C4.5-like (entropy): {acc_1:.4f}")
print(f"CART (gini): {acc_2:.4f}")
print(f"Random Forest: {acc_3:.4f}")

# =====================================================
# 12) Подробный отчет по модели 1
# =====================================================
print("\n" + "=" * 60)
print("Отчет по модели 1: C4.5-like")
print("=" * 60)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_1))
print("Classification report:")
print(classification_report(y_test, y_pred_1))

# =====================================================
# 13) Подробный отчет по модели 2
# =====================================================
print("\n" + "=" * 60)
print("Отчет по модели 2: CART")
print("=" * 60)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_2))
print("Classification report:")
print(classification_report(y_test, y_pred_2))

# =====================================================
# 14) Подробный отчет по модели 3
# =====================================================
print("\n" + "=" * 60)
print("Отчет по модели 3: Random Forest")
print("=" * 60)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_3))
print("Classification report:")
print(classification_report(y_test, y_pred_3))

# =====================================================
# 15) Сравнение результатов
# =====================================================
results = pd.DataFrame({
    "Модель": ["C4.5-like", "CART", "Random Forest"],
    "Accuracy": [acc_1, acc_2, acc_3]})

results = results.sort_values(by="Accuracy", ascending=False)

print("\n" + "=" * 60)
print("Итоговое сравнение моделей")
print("=" * 60)
print(results)

# =====================================================
# 16) Визуализация сравнения
# =====================================================
plt.figure(figsize=(8, 5))
plt.bar(results["Модель"], results["Accuracy"])
plt.title("Сравнение accuracy моделей")
plt.xlabel("Модель")
plt.ylabel("Accuracy")
plt.show()

# =====================================================
# 17) Финальный вывод
# =====================================================
best_model = results.iloc[0]["Модель"]
best_accuracy = results.iloc[0]["Accuracy"]

print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)
print(f"Лучшая модель: {best_model}")
print(f"Лучшая accuracy: {best_accuracy:.4f}")
