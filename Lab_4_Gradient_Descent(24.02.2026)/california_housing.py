import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# 2) Загрузка данных California Housing
# =====================================================
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print("Shape исходных данных:", df.shape)
print(df.head())
print(df.describe())


# =====================================================
# 3) Удаление строк с обрезанным значением 5.00001
# =====================================================
# В данных есть искусственно обрезанные значения (цензурирование) — все дома
# с реальной ценой выше 500 000 были заменены на 500 001. Таких записей 965.
# Чтобы модель не училась на этих «обрезанных» данных, удаляем их.
count_censored = (df['MedHouseVal'] == 5.00001).sum()
print(f"Строк со значением 5.00001 до удаления: {count_censored}")
df = df[df['MedHouseVal'] != 5.00001]
print("Shape после удаления обрезанных значений:", df.shape)


# =====================================================
# 4) Логарифмирование целевой переменной
# =====================================================
# Распределение цен на жильё сильно скошено вправо (long-tail).
# Логарифмирование делает распределение более симметричным,
# что улучшает сходимость градиентного спуска и качество модели.
df['MedHouseVal'] = np.log(df['MedHouseVal'])


# =====================================================
# 5) Удаление признаков по рекомендации
# =====================================================
# Опираемся на код Андрея: удаляем AveBedrms (сильно коррелирует с AveRooms)
# и Longitude (возможно, признак не несёт полезной информации).
df = df.drop(['AveBedrms', 'Longitude'], axis=1)


# =====================================================
# 6) Добавление квадратичных признаков
# =====================================================
# Для учёта возможных нелинейных зависимостей добавляем квадраты
# некоторых признаков. Это позволит линейной модели аппроксимировать
# квадратичные тренды.
df['HouseAge_sq'] = df['HouseAge'] ** 2
df['Population_sq'] = df['Population'] ** 2
df['MedInc_sq'] = df['MedInc'] ** 2


# =====================================================
# 7) Подготовка данных для обучения
# =====================================================
X = df.drop('MedHouseVal', axis=1).values  # признаки
y = df['MedHouseVal'].values.reshape(-1, 1)  # целевая (уже логарифмирована)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Масштабирование признаков (StandardScaler) — критически важно для
# градиентного спуска, особенно при наличии квадратичных признаков.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Конвертация в тензоры PyTorch и перенос на выбранное устройство
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)


# =====================================================
# 8) Создание модели
# =====================================================
# Простая линейная регрессия: один линейный слой без скрытых слоёв.
model = nn.Linear(X_train.shape[1], 1).to(device)

# Функция потерь — среднеквадратичная ошибка (MSE)
criterion = nn.MSELoss()

# Оптимизатор Adam с L2-регуляризацией (weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)


# =====================================================
# 9) Цикл обучения (Batch Gradient Descent)
# =====================================================
epochs = 500
history = []  # список для сохранения значений потерь на каждой эпохе

model.train()
for epoch in range(epochs):
    # Обнуляем градиенты
    optimizer.zero_grad()

    # Прямой проход: предсказание модели
    y_pred = model(X_train_t)

    # Вычисление потерь
    loss = criterion(y_pred, y_train_t)

    # Обратный проход: вычисление градиентов
    loss.backward()

    # Обновление весов
    optimizer.step()

    # Сохраняем значение потерь для последующего построения графика
    history.append(loss.item())

    # Печатаем прогресс каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nОбучение завершено.")


# =====================================================
# 10) Финальная визуализация: 4 графика в одной фигуре
# =====================================================
# Получаем предсказания на тестовой выборке
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t).cpu().numpy()
    y_test_actual = y_test_t.cpu().numpy()

# Остатки = факт - предсказание
residuals = y_test_actual - y_test_pred

# Создаём фигуру 2x2 (четыре графика)
plt.figure(figsize=(16, 12))  # широкое полотно, чтобы вместить тепловую карту

# --- 1) График процесса обучения ---
plt.subplot(2, 2, 1)
plt.plot(history)
plt.title("Процесс обучения (MSE)")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.grid(True)

# --- 2) Анализ остатков (Residuals vs Predicted) ---
plt.subplot(2, 2, 2)
plt.scatter(y_test_pred, residuals, alpha=0.3, color='teal')
plt.axhline(0, color='red', linestyle='--')
plt.title("Анализ остатков")
plt.xlabel("Предсказанное значение (log)")
plt.ylabel("Остатки")

# --- 3) Гистограмма распределения остатков ---
plt.subplot(2, 2, 3)
sns.histplot(residuals, kde=True, bins=30)
plt.title("Распределение остатков")
plt.xlabel("Остаток")
plt.ylabel("Частота")

# --- 4) Корреляционная матрица после преобразований ---
plt.subplot(2, 2, 4)
corr = df.corr()  # вычисляем корреляцию для всего DataFrame (уже с преобразованиями)
# Тепловая карта с уменьшенным шрифтом аннотаций, чтобы подписи не налезали
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
plt.title("Корреляционная матрица")

plt.tight_layout()  # автоматически подгоняет расстояния между графиками
plt.show()