filename = 'text_read.txt'

with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()
num_lines = len(lines)

words = text.split()
num_words = len(words)

letters = []
for simbol in text:
    if 'A' <= simbol.upper() <= 'Z':
        letters.append(simbol)

num_letters = len(letters)

print("Количество строк:", num_lines)
print("Количество слов:", num_words)
print("Количество букв:", num_letters)