from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

########################################################################################################################
# Таргет: Купит ли клиент товар
df1 = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50],
    'ID_System': [np.nan, 102, np.nan, 105, np.nan, 107],
    'Target': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes']})

df1 = df1.drop(columns=['ID_System'])                      # удаляем идентификатор, ID не влияет на предсказание купит или нет
df1["Target"] = df1["Target"].replace({"No": 0, "Yes": 1}) # таргет бинарный (купит/не купит) -> кодируем 0/1 (удобный формат для классификации)
print(df1)
########################################################################################################################

########################################################################################################################
# Таргет: Уровень подписки (Basic < Silver < Gold — с порядком)
df2 = pd.DataFrame({
    "City": ["Moscow", "Moscow", "London", "Moscow", np.nan, "Moscow", "London"],
    "Age": [20, 25, 30, 35, 40, 45, 50],
    "Target": ["Basic", "Basic", "Silver", "Silver", "Gold", "Gold", "Gold"]})
# Кодируем порядковый таргет, сохраняя порядок классов
df2["Target"] = df2["Target"].replace({"Basic": 0, "Silver": 1, "Gold": 2}) # Так как явно есть порядок можно закодировать по возрастанию

X = df2.drop(columns=["Target"])
y = df2["Target"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=10)

imp = SimpleImputer(strategy="constant", fill_value="Unknown") # Меняем на Unknown чтобы не угадывать
X_train2[["City"]] = imp.fit_transform(X_train2[["City"]])
X_test2[["City"]] = imp.transform(X_test2[["City"]])

print("X_train:\n", X_train2, "\n")
print("X_test:\n", X_test2)
########################################################################################################################

########################################################################################################################
# Таргет: Группа здоровья (A < B < C — с порядком)
df3 = pd.DataFrame({
    'Pulse': [70, 72, 75, np.nan, 68, 71, 73, 74],
    'Temp': [36.6, 36.7, 36.8, 36.6, 36.9, 36.6, 36.7, 36.8],
    'Target': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'C']})
df3["Target"] = df3["Target"].replace({"A": 0, "B": 1, "C": 2})  # Так как есть порядок A < B < C, кодируем по возрастанию

X = df3.drop(columns=["Target"])
y = df3["Target"]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=10)

imp = SimpleImputer(strategy="median")  # Pulse — числовой признак, медиана устойчива к выбросам
X_train3[["Pulse"]] = imp.fit_transform(X_train3[["Pulse"]])
X_test3[["Pulse"]] = imp.transform(X_test3[["Pulse"]])

print("X_train:\n", X_train3, "\n")
print("X_test:\n", X_test3)
########################################################################################################################

########################################################################################################################
# Таргет: Прошел проверку безопасности (Да/Нет)
df4 = pd.DataFrame({
    "Days_Since_Last_Incident": [10, 5, 20, np.nan, 15, 30],
    "Risk_Score": [0.1, 0.2, 0.1, 0.4, 0.2, 0.1],
    "Target": ["Safe", "Safe", "Warning", "Safe", "Safe", "Warning"]})

df4["Target"] = df4["Target"].replace({"Safe": 0, "Warning": 1})  # Таргет бинарный (без порядка) -> кодируем 0/1

X = df4.drop(columns=["Target"])
y = df4["Target"]

X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=0.3, random_state=10)

imp = SimpleImputer(strategy="median")  # Days_Since_Last_Incident — числовой признак, медиана устойчива к выбросам
X_train4[["Days_Since_Last_Incident"]] = imp.fit_transform(X_train4[["Days_Since_Last_Incident"]])
X_test4[["Days_Since_Last_Incident"]] = imp.transform(X_test4[["Days_Since_Last_Incident"]])

print("X_train:\n", X_train4, "\n")
print("X_test:\n", X_test4)
########################################################################################################################

########################################################################################################################
# Таргет: Кредитный рейтинг (Low < High — с порядком)
df5 = pd.DataFrame({
    "Bonus_Points": [100, 500, np.nan, 200, np.nan, 800],
    "Salary_K": [50, 100, 40, 120, 30, 150],
    "Target": ["Low", "High", "Low", "High", "Low", "High"]})

df5["Target"] = df5["Target"].replace({"Low": 0, "High": 1})  # Есть порядок Low < High -> кодируем 0/1

X = df5.drop(columns=["Target"])
y = df5["Target"]

X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size=0.3, random_state=10)

imp = SimpleImputer(strategy="median")  # Bonus_Points — числовой признак, медиана устойчива к выбросам
X_train5[["Bonus_Points"]] = imp.fit_transform(X_train5[["Bonus_Points"]])
X_test5[["Bonus_Points"]] = imp.transform(X_test5[["Bonus_Points"]])

print("X_train:\n", X_train5, "\n")
print("X_test:\n", X_test5)
########################################################################################################################
