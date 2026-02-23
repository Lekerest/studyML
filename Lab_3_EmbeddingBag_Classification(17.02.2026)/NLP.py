import re
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset


# =====================================================
# 1) Выбор устройства
# =====================================================
# Если есть NVIDIA GPU → используем CUDA
# Если Mac M-серия → используем MPS
# Иначе fallback на CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# =====================================================
# 2) Загрузка датасета AG News
# =====================================================
# Датасет содержит 4 категории новостей:
# World, Sports, Business, Sci/Tech
dataset = load_dataset("ag_news")


# =====================================================
# 3) Токенизация и построение словаря
# =====================================================
# Токенизация: разбиваем текст на слова + добавляем биграммы
def tokenizer(text: str):
    tokens = re.findall(r"\w+", text.lower())

    # добавляю биграммы, чтобы модель видела пары слов
    bigrams = [
        tokens[i] + "_" + tokens[i + 1]
        for i in range(len(tokens) - 1)
    ]

    return tokens + bigrams


# Считаем частоту слов по train-части
counter = Counter()
for text in dataset["train"]["text"]:
    counter.update(tokenizer(text))


# Создаём словарь
# <pad> — токен для выравнивания (на будущее)
# <unk> — неизвестные слова
vocab = {"<pad>": 0, "<unk>": 1}

# Оставляю только слова, которые встречаются > 5 раз
for word, freq in counter.items():
    if freq > 5:
        vocab[word] = len(vocab)

vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# Преобразование текста в индексы
def text_pipeline(text):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]


# Преобразование метки в int
def label_pipeline(label):
    return int(label)


# =====================================================
# 4) Collate функция для EmbeddingBag
# =====================================================
# EmbeddingBag требует:
# - один длинный тензор всех слов
# - offsets — где начинается каждый текст
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]

    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))

        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    return label_list.to(device), text_list.to(device), offsets.to(device)


# =====================================================
# 5) Модель
# =====================================================
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()

        # Каждый токен превращается в вектор размерности embed_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")

        # Dropout добавляю для регуляризации (чтобы модель не переобучалась)
        self.dropout = nn.Dropout(0.5)

        # Линейный слой для классификации
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.dropout(embedded)
        return self.fc(embedded)


num_class = 4

model = TextClassificationModel(
    vocab_size=vocab_size,
    embed_dim=128,
    num_class=num_class
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =====================================================
# 6) Обучение (1 эпоха)
# =====================================================
train_data = list(zip(dataset["train"]["label"], dataset["train"]["text"]))

dataloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch
)

model.train()

for i, (labels, texts, offsets) in enumerate(dataloader):
    optimizer.zero_grad()

    output = model(texts, offsets)
    loss = criterion(output, labels)

    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"iter={i}, loss={loss.item():.4f}")

print("\nTraining finished.")


# =====================================================
# 7) Предсказание
# =====================================================
ag_news_label = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def predict(text: str):
    model.eval()
    with torch.no_grad():
        tokens = text_pipeline(text)
        text_tensor = torch.tensor(tokens, dtype=torch.int64).to(device)
        offsets = torch.tensor([0]).to(device)

        output = model(text_tensor, offsets)
        return ag_news_label[int(output.argmax(1).item())]


# =====================================================
# 8) Проверка на тестовых примерах
# =====================================================

test_examples = [
    ("The team won the championship after a late goal in the final minute.", "Sports"),
    ("Global stocks fell as investors worried about the new economic policy.", "Business"),
    ("New software update fixes security bugs in popular web browsers.", "Sci/Tech"),
    ("The prime minister met with leaders to discuss the international treaty.", "World"),
    ("The quarterback signed a record-breaking contract this season.", "Sports"),
    ("Oil prices surged following the conflict in the Middle East.", "Business"),
    ("Scientists discovered a new planet orbiting a distant star.", "Sci/Tech"),
    ("The election results caused protests in several major cities.", "World"),
    ("Apple stocks reached a new high after the launch of the new iPhone.", "Business"),
    ("The president attended the Olympic games opening ceremony in Paris.", "World"),
    ("A new AI startup raised 100 million dollars in its first funding round.", "Business"),
    ("FIFA is investigating a corruption scandal involving high-ranking officials.", "Sports"),
    ("The court ruled that the tech giant must pay a massive fine for monopoly.", "World"),
    ("The stadium was packed for the final match of the world cup.", "Sports"),
    ("New battery technology could double the range of electric vehicles.", "Sci/Tech"),
    ("The central bank raised interest rates to combat rising inflation.", "Business"),
    ("SpaceX successfully landed another rocket on the floating platform.", "Sci/Tech"),
    ("NASA is collaborating with private companies for the next moon mission.", "Sci/Tech"),
    ("The star player was injured during the training session on Tuesday.", "Sports"),
    ("The trade agreement between the two nations will reduce import taxes.", "World"),
]

print(f"\n{'PREDICTION':<12} | {'ACTUAL':<10} | {'RESULT':<7} | TEXT")
print("-" * 100)

wrong = []

for text, actual in test_examples:
    predicted = predict(text)
    result = "OK" if predicted == actual else "WRONG"

    print(f"{predicted:<12} | {actual:<10} | {result:<7} | {text}")

    if result == "WRONG":
        wrong.append((text, actual, predicted))

print("\nWrong examples count:", len(wrong))

for text, actual, predicted in wrong:
    print(f"\nACTUAL={actual}  PREDICTED={predicted}\n{text}")