filename = 'read.txt'

with open(filename, 'r', encoding='utf-8') as file:
    content = file.read().split()

numbers = []

for num in content:
    numbers.append(int(num))
total = sum(numbers)

print(total)