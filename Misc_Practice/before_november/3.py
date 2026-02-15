import random

filename = input()

with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    random_line = random.choice(lines)
    print(random_line.rstrip())