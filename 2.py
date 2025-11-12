filename = input()

with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    print(lines[-2].rstrip())