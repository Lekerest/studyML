import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

# ============================================================================
# 1) Загрузка данных Breast Cancer
# ============================================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 0 = malignant_zlo (злокачественная)
# 1 = benign_dobro (доброкачественная)

print("Shape исходных данных:", X.shape)
print("Первые 5 строк:")
print(X.head())
print("\nРаспределение классов:")
print(pd.Series(y).value_counts().sort_index())
print("\nНазвания классов:", data.target_names)

# ============================================================================
# 2) Проверка пропусков
# ============================================================================
print("\nКоличество пропусков по признакам:")
print(X.isnull().sum())

# ============================================================================
# 3) Train / Test split
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\nShape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# ============================================================================
# 4) Масштабирование признаков
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5) Базовая модель
# ============================================================================
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# ============================================================================
# 6) Предсказание
# ============================================================================
y_pred = model.predict(X_test_scaled)

# ============================================================================
# 7) Базовая оценка
# ============================================================================
print("\n" + "=" * 60)
print("БАЗОВАЯ МОДЕЛЬ ИЗ НОУТБУКА")
print("=" * 60)
print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nДетальный отчет по метрикам:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ============================================================================
# 8) Интерпретация весов
# ============================================================================
weights = pd.Series(model.coef_[0], index=data.feature_names)
top_weights = weights.sort_values(ascending=False).head(5)
print("\nТоп-5 признаков, влияющих на решение о доброкачественности:")
print(top_weights)

# ============================================================================
# 9) Подготовка таргета под ТЗ
# ============================================================================
# В ТЗ важно минимизировать FN именно для malignant_zlo.
# Поэтому делаем malignant_zlo = 1, benign_dobro = 0.
y_malignant_zlo = (data.target == 0).astype(int)

print("\n" + "=" * 60)
print("ПОДГОТОВКА ТАРГЕТА ПОД ТЗ")
print("=" * 60)
print("Теперь positive class = malignant_zlo")
print("1 = malignant_zlo, 0 = benign_dobro")
print("Количество malignant_zlo:", y_malignant_zlo.sum())
print("Количество benign_dobro:", len(y_malignant_zlo) - y_malignant_zlo.sum())

# ============================================================================
# 10) Новый split для основной задачи
# ============================================================================
# Используем stratify, чтобы сохранить баланс классов.
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y_malignant_zlo, test_size=0.25, random_state=42, stratify=y_malignant_zlo)

# Дополнительно делим train на train/val для честного подбора threshold
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print("\n" + "=" * 60)
print("РАЗДЕЛЕНИЕ ДАННЫХ ДЛЯ ОСНОВНОЙ ЗАДАЧИ")
print("=" * 60)
print("X_train_main:", X_train_main.shape)
print("X_val:", X_val.shape)
print("X_test_final:", X_test_final.shape)

# ============================================================================
# 11) Масштабирование для основной задачи
# ============================================================================
scaler_main = StandardScaler()
X_train_main_scaled = scaler_main.fit_transform(X_train_main)
X_val_scaled = scaler_main.transform(X_val)
X_test_final_scaled = scaler_main.transform(X_test_final)

# ============================================================================
# 12) Базовая стратегия под ТЗ
# ============================================================================
# Выбираем LogisticRegression с L2-регуляризацией.
# class_weight='balanced' помогает лучше учитывать malignant_zlo.
base_recall_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    class_weight='balanced',
    max_iter=3000,
    random_state=42
)
base_recall_model.fit(X_train_main_scaled, y_train_main)

# ============================================================================
# 13) Оценка baseline на validation при стандартном threshold = 0.5
# ============================================================================
val_proba_base = base_recall_model.predict_proba(X_val_scaled)[:, 1]
val_pred_base = (val_proba_base >= 0.5).astype(int)

recall_base = recall_score(y_val, val_pred_base)
acc_base = accuracy_score(y_val, val_pred_base)
cm_base = confusion_matrix(y_val, val_pred_base)

print("\n" + "=" * 60)
print("BASELINE ДЛЯ ОСНОВНОЙ ЗАДАЧИ")
print("=" * 60)
print("Стандартный threshold = 0.5")
print(f"Recall malignant_zlo: {recall_base:.4f}")
print(f"Accuracy: {acc_base:.4f}")
print("Confusion matrix:")
print(cm_base)

# ============================================================================
# 14) Работа с признаками
# ============================================================================
# Используем абсолютные значения коэффициентов baseline-модели
# и оставляем наиболее важные признаки.
coef_abs = np.abs(base_recall_model.coef_[0])
feature_importance = pd.Series(coef_abs, index=X.columns).sort_values(ascending=False)

top_10_features = feature_importance.head(10).index.tolist()

print("\n" + "=" * 60)
print("ОТБОР ПРИЗНАКОВ")
print("=" * 60)
print("Топ-10 признаков по абсолютным коэффициентам:")
for i, feature in enumerate(top_10_features, start=1):
    print(f"{i}. {feature}")

X_top10 = X[top_10_features]

X_train_full_top10, X_test_final_top10, y_train_full_top10, y_test_final_top10 = train_test_split(
    X_top10, y_malignant_zlo, test_size=0.25, random_state=42, stratify=y_malignant_zlo
)

X_train_main_top10, X_val_top10, y_train_main_top10, y_val_top10 = train_test_split(
    X_train_full_top10, y_train_full_top10, test_size=0.2, random_state=42, stratify=y_train_full_top10
)

scaler_top10 = StandardScaler()
X_train_main_top10_scaled = scaler_top10.fit_transform(X_train_main_top10)
X_val_top10_scaled = scaler_top10.transform(X_val_top10)
X_test_final_top10_scaled = scaler_top10.transform(X_test_final_top10)

top10_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    class_weight='balanced',
    max_iter=3000,
    random_state=42
)
top10_model.fit(X_train_main_top10_scaled, y_train_main_top10)

val_proba_top10 = top10_model.predict_proba(X_val_top10_scaled)[:, 1]
val_pred_top10 = (val_proba_top10 >= 0.5).astype(int)

recall_top10 = recall_score(y_val_top10, val_pred_top10)
acc_top10 = accuracy_score(y_val_top10, val_pred_top10)
cm_top10 = confusion_matrix(y_val_top10, val_pred_top10)

print("\nМодель только на top-10 признаках")
print(f"Recall malignant_zlo: {recall_top10:.4f}")
print(f"Accuracy: {acc_top10:.4f}")
print("Confusion matrix:")
print(cm_top10)

# ============================================================================
# 15) Подбор threshold для минимизации FN
# ============================================================================
# Ищем порог, который максимизирует Recall malignant_zlo.
# При одинаковом Recall выбираем тот, где Accuracy выше.
thresholds = np.arange(0.10, 0.91, 0.05)

results_all_features = []
results_top10 = []

for threshold in thresholds:
    pred_all = (val_proba_base >= threshold).astype(int)
    recall_all = recall_score(y_val, pred_all)
    acc_all = accuracy_score(y_val, pred_all)
    cm_all = confusion_matrix(y_val, pred_all)
    fn_all = cm_all[1, 0]

    results_all_features.append({
        'threshold': threshold,
        'recall': recall_all,
        'accuracy': acc_all,
        'fn': fn_all
    })

    pred_top = (val_proba_top10 >= threshold).astype(int)
    recall_top = recall_score(y_val_top10, pred_top)
    acc_top = accuracy_score(y_val_top10, pred_top)
    cm_top = confusion_matrix(y_val_top10, pred_top)
    fn_top = cm_top[1, 0]

    results_top10.append({
        'threshold': threshold,
        'recall': recall_top,
        'accuracy': acc_top,
        'fn': fn_top
    })

df_all = pd.DataFrame(results_all_features)
df_top10 = pd.DataFrame(results_top10)

best_all = df_all.sort_values(by=['recall', 'accuracy'], ascending=[False, False]).iloc[0]
best_top10 = df_top10.sort_values(by=['recall', 'accuracy'], ascending=[False, False]).iloc[0]

print("\n" + "=" * 60)
print("ПОДБОР THRESHOLD")
print("=" * 60)
print("Лучший вариант для модели на всех признаках:")
print(best_all)

print("\nЛучший вариант для модели на top-10 признаках:")
print(best_top10)

# ============================================================================
# 16) Выбор лучшей стратегии
# ============================================================================
# Сначала сравниваем Recall.
# Если Recall одинаковый, берем вариант с большей Accuracy.
if best_all['recall'] > best_top10['recall']:
    best_variant_name = 'all_features'
elif best_all['recall'] < best_top10['recall']:
    best_variant_name = 'top10_features'
else:
    best_variant_name = 'all_features' if best_all['accuracy'] >= best_top10['accuracy'] else 'top10_features'

print("\n" + "=" * 60)
print("ВЫБОР ЛУЧШЕЙ СТРАТЕГИИ")
print("=" * 60)
print("Лучшая стратегия:", best_variant_name)

# ============================================================================
# 17) Переобучение лучшей модели на train_full и финальная проверка на test
# ============================================================================
if best_variant_name == 'all_features':
    best_threshold = float(best_all['threshold'])

    scaler_final = StandardScaler()
    X_train_full_scaled = scaler_final.fit_transform(X_train_full)
    X_test_final_scaled = scaler_final.transform(X_test_final)

    final_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        max_iter=3000,
        random_state=42
    )
    final_model.fit(X_train_full_scaled, y_train_full)

    test_proba = final_model.predict_proba(X_test_final_scaled)[:, 1]
    y_test_pred_final = (test_proba >= best_threshold).astype(int)

    used_features = list(X.columns)

else:
    best_threshold = float(best_top10['threshold'])

    scaler_final = StandardScaler()
    X_train_full_top10_scaled = scaler_final.fit_transform(X_train_full_top10)
    X_test_final_top10_scaled = scaler_final.transform(X_test_final_top10)

    final_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        max_iter=3000,
        random_state=42
    )
    final_model.fit(X_train_full_top10_scaled, y_train_full_top10)

    test_proba = final_model.predict_proba(X_test_final_top10_scaled)[:, 1]
    y_test_pred_final = (test_proba >= best_threshold).astype(int)

    used_features = top_10_features

# ============================================================================
# 18) Финальная оценка на test
# ============================================================================
final_cm = confusion_matrix(y_test_final, y_test_pred_final)
final_recall = recall_score(y_test_final, y_test_pred_final)
final_accuracy = accuracy_score(y_test_final, y_test_pred_final)

print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ МОДЕЛЬ НА TEST")
print("=" * 60)
print("Использованные признаки:")
for feature in used_features:
    print("-", feature)

print(f"\nИспользованный threshold: {best_threshold:.2f}")
print(f"Final Recall malignant_zlo: {final_recall:.4f}")
print(f"Final Accuracy: {final_accuracy:.4f}")
print("\nConfusion matrix:")
print(final_cm)

print("\nClassification report:")
print(classification_report(
    y_test_final,
    y_test_pred_final,
    target_names=['benign_dobro', 'malignant_zlo']
))

# ============================================================================
# 19) Визуализация confusion matrix + Recall и Accuracy по threshold
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Левый график: Confusion Matrix
sns.heatmap(
    final_cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=['Pred benign_dobro', 'Pred malignant_zlo'],
    yticklabels=['True benign_dobro', 'True malignant_zlo'],
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix — Final Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Правый график: Recall и Accuracy по threshold
axes[1].plot(df_all['threshold'], df_all['recall'], marker='o', label='Recall (all features)')
axes[1].plot(df_all['threshold'], df_all['accuracy'], marker='o', label='Accuracy (all features)')
axes[1].plot(df_top10['threshold'], df_top10['recall'], marker='s', label='Recall (top-10 features)')
axes[1].plot(df_top10['threshold'], df_top10['accuracy'], marker='s', label='Accuracy (top-10 features)')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Recall / Accuracy vs Threshold')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()


# ============================================================================
# 20) Финальный вывод
# ============================================================================
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЙ ВЫВОД")
print("=" * 60)
print("\nИтог:")
print(f"- Лучшая стратегия: {best_variant_name}")
print(f"- Лучший threshold: {best_threshold:.2f}")
print(f"- Финальный Recall malignant_zlo: {final_recall:.4f}")
print(f"- Финальный Accuracy: {final_accuracy:.4f}")
print("Чем выше Recall, тем меньше злокачественных опухолей модель пропускает.")
