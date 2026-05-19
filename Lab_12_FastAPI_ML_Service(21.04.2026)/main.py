import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# 1) Выбор устройства (GPU / MPS / CPU)
# =====================================================
if torch is not None and torch.cuda.is_available():
    device = "cuda"
elif torch is not None and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device:", device)

# =====================================================
# 2) Глобальные переменные приложения
# =====================================================
model = None
target_names = None
model_accuracy = None
model_report = None
model_confusion_matrix = None

# =====================================================
# 3) Модель входных данных Pydantic
# =====================================================
class WineRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100.0,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "od280_od315_of_diluted_wines": 3.4,
                "proline": 1050.0
            }
        }
    )

# =====================================================
# 4) Модель ответа Pydantic
# =====================================================
class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict

# =====================================================
# 5) Подготовка признаков для модели
# =====================================================
def make_features(request: WineRequest):
    features = pd.DataFrame([{
        "alcohol": request.alcohol,
        "malic_acid": request.malic_acid,
        "ash": request.ash,
        "alcalinity_of_ash": request.alcalinity_of_ash,
        "magnesium": request.magnesium,
        "total_phenols": request.total_phenols,
        "flavanoids": request.flavanoids,
        "nonflavanoid_phenols": request.nonflavanoid_phenols,
        "proanthocyanins": request.proanthocyanins,
        "color_intensity": request.color_intensity,
        "hue": request.hue,
        "od280/od315_of_diluted_wines": request.od280_od315_of_diluted_wines,
        "proline": request.proline
    }])

    return features

# =====================================================
# 6) Логирование предсказаний
# =====================================================
def save_prediction_log(predicted_class_name: str):
    with open("prediction_log.txt", "a", encoding="utf-8") as file:
        file.write(f"{datetime.now()} - predicted_class: {predicted_class_name}\n")

# =====================================================
# 7) Обучение модели
# =====================================================
def train_model():
    global model
    global target_names
    global model_accuracy
    global model_report
    global model_confusion_matrix

    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    target_names = list(wine.target_names)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Классы:")
    print(target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Shape X_train:", X_train.shape)
    print("Shape X_test:", X_test.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape y_test:", y_test.shape)

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42))])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_accuracy = accuracy_score(y_test, y_pred)
    model_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True)
    model_confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 60)
    print("Качество модели")
    print("=" * 60)
    print(f"Accuracy: {model_accuracy:.4f}")

# =====================================================
# 8) Lifespan приложения
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("Запуск приложения")
    print("=" * 60)

    train_model()

    yield

    print("\n" + "=" * 60)
    print("Остановка приложения")
    print("=" * 60)

# =====================================================
# 9) Создание FastAPI приложения
# =====================================================
app = FastAPI(
    title="Wine ML Service",
    description="FastAPI сервис для классификации объектов Wine",
    version="1.0",
    lifespan=lifespan)

router = APIRouter(prefix="/api")

# =====================================================
# 10) Главный маршрут
# =====================================================
@app.get("/")
async def root():
    return {
        "message": "Wine ML Service is running",
        "docs": "/docs"
    }

# =====================================================
# 11) Информация о модели
# =====================================================
@router.get("/model/info")
async def model_info():
    return {
        "model": "LogisticRegression",
        "dataset": "Wine",
        "device": device,
        "classes": target_names,
        "accuracy": model_accuracy
    }

# =====================================================
# 12) Метрики модели
# =====================================================
@router.get("/model/metrics")
async def model_metrics():
    return {
        "accuracy": model_accuracy,
        "confusion_matrix": model_confusion_matrix,
        "classification_report": model_report
    }

# =====================================================
# 13) Предсказание класса
# =====================================================
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: WineRequest, background_tasks: BackgroundTasks):
    features = make_features(request)

    predicted_class_id = int(model.predict(features)[0])
    predicted_class_name = target_names[predicted_class_id]

    probabilities_values = model.predict_proba(features)[0]
    probabilities = {}

    for class_name, probability in zip(target_names, probabilities_values):
        probabilities[class_name] = round(float(probability), 4)

    background_tasks.add_task(save_prediction_log, predicted_class_name)

    return JSONResponse(
        status_code=200,
        content={
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": predicted_class_name,
            "probabilities": probabilities
        })

# =====================================================
# 14) Подключение маршрутов
# =====================================================
app.include_router(router)

# =====================================================
# 15) Финальный вывод
# =====================================================
print("\n" + "=" * 60)
print("FastAPI приложение подготовлено")
print("=" * 60)