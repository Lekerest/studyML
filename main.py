n = int(input("Количество строк: "))
lines = [input().strip() for _ in range(n)]

numbers, non_numbers = [], []

for line in lines:
    try:
        numbers.append(float(line))
    except ValueError:
        non_numbers.append(line)

print(f"Сумма: {sum(numbers)}")
print(f"Слова: {sorted(non_numbers)}")