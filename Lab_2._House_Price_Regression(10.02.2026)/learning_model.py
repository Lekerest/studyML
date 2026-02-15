import numpy as np
import pandas as pd

# Модули для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


# =====================================================
# Функции для вычисления метрик
# =====================================================

def rmse_calculate(y_test, y_pred):
    # RMSE — корень из MSE, измеряется в тех же единицах, что и цена
    return np.sqrt(mean_squared_error(y_test, y_pred))


def evaluate_and_print(exp_name, y_test, y_pred, extra_text=None):
    # Вычисляем основные метрики качества модели

    mse = mean_squared_error(y_test, y_pred)        # средняя квадратичная ошибка
    rmse_val = rmse_calculate(y_test, y_pred)       # корень из MSE
    mae = mean_absolute_error(y_test, y_pred)       # средняя абсолютная ошибка
    r2 = r2_score(y_test, y_pred)                   # коэффициент детерминации

    # Вывод результатов
    print(f"\n{exp_name}")
    if extra_text:
        print(extra_text)

    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse_val:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R2   : {r2:.3f}")

    # Возвращаем результаты для итоговой таблицы
    return {"Experiment": exp_name, "MSE": mse, "RMSE": rmse_val, "MAE": mae, "R2": r2}


# =====================================================
# 1) Загрузка и подготовка данных
# =====================================================

# Загружаем датасет
data = "house_price_regression_dataset.csv"
df = pd.read_csv(data)

results = []  # список для хранения результатов всех экспериментов

# Выделяем целевую переменную
target = "House_Price"

# X — признаки (все столбцы кроме цены)
X = df.drop(columns=[target])

# y — целевая переменная (цена)
y = df[target]

# Делим данные на обучающую и тестовую выборку (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

print("Dataset shape:", df.shape)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Смотрим корреляцию признаков с ценой
# Это помогает понять, какие признаки влияют сильнее
corr = df.corr(numeric_only=True)[target].sort_values(key=lambda s: s.abs())
print("\nCorrelation with House_Price (sorted by abs):")
print(corr)


# =====================================================
# Exp1: LinearRegression (без масштабирования)
# =====================================================

# Обучаем обычную линейную регрессию
model_1 = LinearRegression()
model_1.fit(X_train, y_train)

# Делаем предсказания на тесте
y_pred1 = model_1.predict(X_test)

# Оцениваем качество
results.append(evaluate_and_print(
    "Exp1: LinearRegression (no scaling)", y_test, y_pred1))


# =====================================================
# Exp2: StandardScaler + LinearRegression
# =====================================================

# Масштабируем признаки (приводим к среднему 0 и std 1)
scaler2 = StandardScaler()

# ВАЖНО: fit только на train
X_train_scaled2 = scaler2.fit_transform(X_train)
X_test_scaled2 = scaler2.transform(X_test)

# Обучаем модель на масштабированных данных
model_2 = LinearRegression()
model_2.fit(X_train_scaled2, y_train)

# Предсказания
y_pred2 = model_2.predict(X_test_scaled2)

# Оценка
results.append(evaluate_and_print(
    "Exp2: StandardScaler + LinearRegression", y_test, y_pred2))


# =====================================================
# Exp3: Удаляем слабые признаки
# =====================================================

# Убираем признаки с низкой корреляцией
to_drop = ["Num_Bathrooms", "Neighborhood_Quality", "Num_Bedrooms"]

df3 = df.drop(columns=to_drop)

# Формируем новые X и y
X3 = df3.drop(columns=[target])
y3 = df3[target]

# Делим данные
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=10)

# Обучаем модель
model_3 = LinearRegression()
model_3.fit(X3_train, y3_train)

# Предсказания
y_pred3 = model_3.predict(X3_test)

# Оценка
results.append(evaluate_and_print(
    "Exp3: Drop weak features + LinearRegression",
    y3_test, y_pred3,
    extra_text=f"Dropped: {to_drop}"))


# =====================================================
# Exp4: Используем только 2 сильных признака
# =====================================================

# Оставляем только самые сильные признаки
df4 = df[["Square_Footage", "Lot_Size", target]].copy()

X4 = df4.drop(columns=[target])
y4 = df4[target]

X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=10)

model_4 = LinearRegression()
model_4.fit(X4_train, y4_train)

y_pred4 = model_4.predict(X4_test)

results.append(evaluate_and_print(
    "Exp4: Only Square_Footage + Lot_Size",
    y4_test, y_pred4,
    extra_text="Only: Square_Footage, Lot_Size"))


# =====================================================
# Exp5: Ridge + StandardScaler
# =====================================================

# Масштабирование обязательно для Ridge
scaler5 = StandardScaler()
X_train_scaled5 = scaler5.fit_transform(X_train)
X_test_scaled5 = scaler5.transform(X_test)

# Ridge с регуляризацией
alpha = 4
model_5 = Ridge(alpha=alpha)

model_5.fit(X_train_scaled5, y_train)
y_pred5 = model_5.predict(X_test_scaled5)

results.append(evaluate_and_print(
    f"Exp5: StandardScaler + Ridge (alpha={alpha})",
    y_test, y_pred5))


# =====================================================
# Итоговая таблица
# =====================================================

# Собираем результаты всех экспериментов
summary = pd.DataFrame(results).copy()

# Переставляем колонки для удобства
summary = summary[["Experiment", "RMSE", "MAE", "R2", "MSE"]]

# Округляем значения для красивого вывода
summary["MSE"] = summary["MSE"].round(0).astype(int)
summary["RMSE"] = summary["RMSE"].round(0).astype(int)
summary["MAE"] = summary["MAE"].round(0).astype(int)
summary["R2"] = summary["R2"].round(4)

# Сортируем по RMSE (меньше — лучше)
summary = summary.sort_values("RMSE").reset_index(drop=True)

# Выравниваем названия экспериментов
exp_width = 43
summary["Experiment"] = summary["Experiment"].str.ljust(exp_width)

print("\n=== Summary (sorted by RMSE) ===")
print(summary.to_string(index=False))
