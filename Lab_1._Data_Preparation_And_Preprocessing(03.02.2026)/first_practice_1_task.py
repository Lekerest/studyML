import numpy as np                                    # подключаем NumPy для работы с массивами
from sklearn.model_selection import train_test_split  # импортируем функцию разбиения на train/test

X = np.random.randint(10, 101, size=(40, 2))# признаки: 40 строк, 2 столбца, числа 10..100
y = np.random.randint(0, 2, size=(40, 1))   # таргет: 40 строк, 1 столбец, значения 0 или 1

data = np.hstack([X, y])                              # объединяем X и y по столбцам -> получаем (40, 3)

data_features = data[:, :2]                           # берём первые 2 столбца как признаки -> (40, 2)
target = data[:, 2]                                   # берём 3-й столбец как таргет -> (40,)

X_train, X_test, y_train, y_test = train_test_split(  # делим на обучающую и тестовую выборки
    data_features,                             # X: признаки
    target,                                           # y: таргет
    test_size=0.3,                                    # 30% в тест, 70% в обучение
    random_state=10                                   # фиксируем случайность для повторяемого результата
)

print("data shape:", data.shape)                      # выводим форму data (должно быть (40, 3))
print("X_train:", X_train.shape, "y_train:", y_train.shape)  # формы train (28, 2) и (28,)
print("X_test:", X_test.shape, "y_test:", y_test.shape)      # формы test (12, 2) и (12,)
