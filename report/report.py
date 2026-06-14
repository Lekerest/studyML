# =====================================================
# 1) Выбор устройства (GPU / MPS / CPU)
# =====================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import gzip
import struct
import urllib.request
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# =====================================================
# 2) Фиксация seed и папки для результатов
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR = "article_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# 3) Загрузка Fashion-MNIST без torchvision
# =====================================================
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

DATA_DIR = Path("data") / "FashionMNIST" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FASHION_MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
}


def download_if_needed():
    for filename, url in FASHION_MNIST_URLS.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print("Downloading:", filename)
            urllib.request.urlretrieve(url, file_path)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Incorrect image file magic number")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Incorrect label file magic number")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, label


download_if_needed()

train_images = read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")
train_labels = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
test_images = read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
test_labels = read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

train_dataset = FashionMNISTDataset(train_images, train_labels)
test_dataset = FashionMNISTDataset(test_images, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Image shape:", train_dataset[0][0].shape)
print("Classes:", len(class_names))


# =====================================================
# 4) Архитектура сверточной нейронной сети
# =====================================================
class CNNModel(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(CNNModel, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        layers_1 = [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers_1.append(nn.BatchNorm2d(32))
        layers_1.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        layers_2 = [
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers_2.append(nn.BatchNorm2d(64))
        layers_2.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.conv_block = nn.Sequential(
            *layers_1,
            *layers_2
        )

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        ]

        if use_dropout:
            classifier_layers.append(nn.Dropout(p=0.5))

        classifier_layers.append(nn.Linear(128, 10))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


# =====================================================
# 5) Обучение одной модели
# =====================================================
def train_one_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_loss_sum / test_total
        test_accuracy = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_accuracy:.4f}"
        )

    return history


# =====================================================
# 6) Оценка модели на тестовой выборке
# =====================================================
def evaluate_model(model, test_loader):
    model.eval()

    all_labels = []
    all_predictions = []
    test_loss_sum = 0.0
    test_total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            test_loss_sum += loss.item() * images.size(0)
            test_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss = test_loss_sum / test_total
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": np.array(all_labels),
        "predictions": np.array(all_predictions)
    }


# =====================================================
# 7) Обучение всех моделей
# =====================================================
experiments = {
    "CNN": {
        "use_dropout": False,
        "use_batchnorm": False
    },
    "CNN + Dropout": {
        "use_dropout": True,
        "use_batchnorm": False
    },
    "CNN + BatchNorm": {
        "use_dropout": False,
        "use_batchnorm": True
    },
    "CNN + Dropout + BatchNorm": {
        "use_dropout": True,
        "use_batchnorm": True
    }
}

histories = {}
results = []
models = {}
all_eval_results = {}

for model_name, params in experiments.items():
    print("\n" + "=" * 60)
    print("Training model:", model_name)
    print("=" * 60)

    model = CNNModel(
        use_dropout=params["use_dropout"],
        use_batchnorm=params["use_batchnorm"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_one_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS
    )

    eval_result = evaluate_model(model, test_loader)

    histories[model_name] = history
    models[model_name] = model
    all_eval_results[model_name] = eval_result

    results.append({
        "Model": model_name,
        "Accuracy": eval_result["accuracy"],
        "Precision": eval_result["precision"],
        "Recall": eval_result["recall"],
        "F1-score": eval_result["f1_score"],
        "Test loss": eval_result["test_loss"]
    })

    print("\nClassification report:")
    print(classification_report(
        eval_result["labels"],
        eval_result["predictions"],
        target_names=class_names,
        zero_division=0
    ))


# =====================================================
# 8) Итоговая таблица результатов
# =====================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

print("\n" + "=" * 60)
print("Final results:")
print("=" * 60)
print(results_df)

results_csv_path = os.path.join(OUTPUT_DIR, "results.csv")
results_df.to_csv(results_csv_path, index=False)
print("\nSaved:", results_csv_path)

print("\nLaTeX table rows for article:")
for _, row in results_df.iterrows():
    print(
        f"{row['Model']} & "
        f"{row['Accuracy']:.3f} & "
        f"{row['F1-score']:.3f} & "
        f"{row['Test loss']:.3f} \\\\"
    )


# =====================================================
# 9) Графики accuracy и loss для статьи
# =====================================================
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(8, 5))
for model_name, history in histories.items():
    plt.plot(epochs, history["test_accuracy"], marker="o", label=model_name)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
accuracy_path = os.path.join(OUTPUT_DIR, "accuracy_curves.png")
plt.savefig(accuracy_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
for model_name, history in histories.items():
    plt.plot(epochs, history["test_loss"], marker="o", label=model_name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
loss_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
plt.savefig(loss_path, dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for model_name, history in histories.items():
    axes[0].plot(epochs, history["test_accuracy"], marker="o", label=model_name)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].grid(True)
axes[0].legend(fontsize=7)

for model_name, history in histories.items():
    axes[1].plot(epochs, history["test_loss"], marker="o", label=model_name)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
axes[1].legend(fontsize=7)

plt.tight_layout()
training_curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(training_curves_path, dpi=300)
plt.close()

print("Saved:", accuracy_path)
print("Saved:", loss_path)
print("Saved:", training_curves_path)


# =====================================================
# 10) Матрица ошибок для лучшей модели
# =====================================================
best_model_name = results_df.iloc[0]["Model"]
best_eval = all_eval_results[best_model_name]

cm = confusion_matrix(best_eval["labels"], best_eval["predictions"])

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title(f"Confusion matrix: {best_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
plt.yticks(np.arange(len(class_names)), class_names, fontsize=8)
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

plt.tight_layout()
confusion_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(confusion_path, dpi=300)
plt.close()

print("Best model:", best_model_name)
print("Saved:", confusion_path)


# =====================================================
# 11) Сохранение лучшей модели
# =====================================================
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
torch.save(models[best_model_name].state_dict(), best_model_path)
print("Saved:", best_model_path)


# =====================================================
# 12) Итоговый вывод
# =====================================================
print("\n" + "=" * 60)
print("Итоговый вывод")
print("=" * 60)
print(f"Лучшая модель: {best_model_name}")
print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"F1-score: {results_df.iloc[0]['F1-score']:.4f}")
print(f"Test loss: {results_df.iloc[0]['Test loss']:.4f}")
print("\nФайлы для Overleaf находятся в папке:", OUTPUT_DIR)
print("Нужно загрузить в Overleaf training_curves.png и confusion_matrix.png")
