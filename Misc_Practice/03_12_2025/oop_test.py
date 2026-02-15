class Weather:
    planet = "Земля"
    unit_temp = "°C"
    unit_wind = "м/с"

    def __init__(self, city, temp, wind, condition):
        self.city = city
        self.temp = temp
        self.wind = wind
        self.condition = condition

w1 = Weather("Амстердам", 6, 5, "облачно")
w2 = Weather("Лиссабон", 15, 3, "солнечно")
w3 = Weather("Рейкьявик", 1, 8, "снег")

w1.humidity = 82         # только у w1
w2.uv_index = 4          # только у w2

print("\nАтрибуты класса доступны всем экземплярам")
for w in (w1, w2, w3):
    print(w.city, "->", w.planet, w.unit_temp, w.unit_wind)

print("\n=== Проверяем уникальные атрибуты (есть/нет) ===")
print("w1.humidity:", hasattr(w1, "humidity"), "| w2.humidity:", hasattr(w2, "humidity"))
print("w2.uv_index:", hasattr(w2, "uv_index"), "| w3.uv_index:", hasattr(w3, "uv_index"))

print("\nСловарь атрибутов экземпляра (уникальные + заданные в __init__)")
print("w1.__dict__:", w1.__dict__)
print("w2.__dict__:", w2.__dict__)
print("w3.__dict__:", w3.__dict__)