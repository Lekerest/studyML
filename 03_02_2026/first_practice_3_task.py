from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

########################################################################################################################
# MinMaxScaler (Нормализация)
# таргет: уровень премии (Low < Medium < High — с порядком)
df6 = pd.DataFrame({
    "Completion_Pct": [10, 25, 45, 50, 75, 85, 95, 100],
    "Experience_Years": [1, 2, 3, 4, 5, 6, 7, 8],
    "Target": ["Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High"]})

df6["Target"] = df6["Target"].replace({"Low": 0, "Medium": 1, "High": 2})  # Есть порядок Low < Medium < High -> кодируем 0/1/2

X = df6.drop(columns=["Target"])
y = df6["Target"]

X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()  # MinMaxScaler нормализует признаки в диапазон [0, 1]
X_train6[["Completion_Pct", "Experience_Years"]] = scaler.fit_transform(X_train6[["Completion_Pct", "Experience_Years"]])
X_test6[["Completion_Pct", "Experience_Years"]] = scaler.transform(X_test6[["Completion_Pct", "Experience_Years"]])
print("X_train:\n", X_train6, "\n")
print("X_test:\n", X_test6)
########################################################################################################################

########################################################################################################################
# таргет: одобрение кредита (Yes/No)
df7 = pd.DataFrame({
    "Income_K": [30, 35, 40, 45, 50, 42, 38, 1000],
    "Credit_Score": [600, 620, 640, 610, 650, 630, 615, 800],
    "Target": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "Yes"]})

df7["Target"] = df7["Target"].replace({"No": 0, "Yes": 1})  # Таргет бинарный -> кодируем 0/1

X = df7.drop(columns=["Target"])
y = df7["Target"]

X_train7, X_test7, y_train7, y_test7 = train_test_split(X, y, test_size=0.3, random_state=10)

scaler = StandardScaler()  # StandardScaler стандартизирует признаки (x - mean) / std
X_train7[["Income_K", "Credit_Score"]] = scaler.fit_transform(X_train7[["Income_K", "Credit_Score"]])
X_test7[["Income_K", "Credit_Score"]] = scaler.transform(X_test7[["Income_K", "Credit_Score"]])

print("X_train:\n", X_train7, "\n")
print("X_test:\n", X_test7)
########################################################################################################################
