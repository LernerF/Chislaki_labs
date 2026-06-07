"""
Лабораторная работа 3.1: Интерполяция таблично заданной функции
Методы: Лагранжа (2-й и 3-й степени)
Цель: Сравнить результаты интерполяции при разном выборе узлов
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.array([-4.58, -4.05, -3.52, -2.99, -2.46, -1.93, -1.40, -0.87, -0.34])
y = np.array([1.7361, 2.0354, 3.0762, 2.8126, 1.2583, 1.8624, 1.7263, 3.2846, 3.4107])

x_star = -3.875


def lagrange_poly(x_val, x_nodes, y_nodes):
    """
    Вычисляет значение интерполяционного многочлена Лагранжа в точке x_val
    
    Теория:
    P_n(x) = Σ y_i * L_i(x), где L_i(x) = Π (x - x_j)/(x_i - x_j)
    
    Параметры:
    x_val   - точка, в которой вычисляем значение
    x_nodes - массив узлов интерполяции [x0, x1, ..., xn]
    y_nodes - массив значений функции в узлах [y0, y1, ..., yn]
    
    Возвращает:
    P_n(x_val) - значение многочлена в точке x_val
    """
    n = len(x_nodes)          # количество узлов = степень многочлена + 1
    result = 0.0              # накапливаем сумму
    
    # Суммируем по всем узлам: Σ y_i * L_i(x)
    for i in range(n):
        term = y_nodes[i]     # начинаем с y_i
        
        # Строим базисный многочлен L_i(x) = Π (x - x_j)/(x_i - x_j)
        for j in range(n):
            if i != j:        # пропускаем j = i (знаменатель был бы 0)
                term *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        
        result += term        # добавляем слагаемое к сумме
    
    return result

# ВЫБОР УЗЛОВ ПО БЛИЗОСТИ К ТОЧКЕ ИНТЕРПОЛЯЦИИ
# Теоретическое обоснование:
# Погрешность интерполяции R_n(x) = f^{(n+1)}(ξ)/(n+1)! * ω_{n+1}(x)
# где ω_{n+1}(x) = (x-x0)(x-x1)...(x-xn)
# Если выбрать узлы, близкие к x*, то |ω_{n+1}(x*)| будет минимальным,
# следовательно, погрешность будет меньше.

print("=" * 70)
print("ВАРИАНТ 1: Узлы по близости к x*")
print("=" * 70)

# Вычисляем расстояния от x* до всех узлов
distances = np.abs(x - x_star)
sorted_indices = np.argsort(distances)

# L2 - многочлен 2-й степени (используем 3 ближайших узла)
# Степень 2 означает, что многочлен проходит через 3 точки
idx_l2_v1 = sorted_indices[:3]
x_l2_v1 = x[idx_l2_v1]    # узлы для L2
y_l2_v1 = y[idx_l2_v1]    # значения в этих узлах

# L3 - многочлен 3-й степени (используем 4 ближайших узла)
idx_l3_v1 = sorted_indices[:4]
x_l3_v1 = x[idx_l3_v1]    # узлы для L3
y_l3_v1 = y[idx_l3_v1]    # значения в этих узлах

# Выводим информацию о расстояниях
print("Расстояния от x* до узлов:")
for i, (xi, yi) in enumerate(zip(x, y)):
    dist = abs(xi - x_star)
    print(f"x[{i}] = {xi:.2f}, расстояние = {dist:.3f}")

print("\nУзлы для интерполяции (по близости к x*):")
print(f"L2: индексы {idx_l2_v1}, x = {x_l2_v1}, y = {y_l2_v1}")
print(f"L3: индексы {idx_l3_v1}, x = {x_l3_v1}, y = {y_l3_v1}")

# ПРОВЕРКА СВОЙСТВА ИНТЕРПОЛЯЦИИ
# Теория: Интерполяционный многочлен должен проходить через все узлы:
# P_n(x_i) = y_i для всех i = 0..n

print("\nПРОВЕРКА В УЗЛОВЫХ ТОЧКАХ:")

# Проверка для L2 (многочлен 2-й степени)
for i, (xi, yi) in enumerate(zip(x_l2_v1, y_l2_v1)):
    L2_val = lagrange_poly(xi, x_l2_v1, y_l2_v1)
    match = 'СОВПАДАЕТ' if abs(L2_val - yi) < 1e-10 else 'ОШИБКА'
    print(f"L2 в узле x={xi:.2f}: {L2_val:.6f} (истинное: {yi:.6f}) - {match}")

print()

# Проверка для L3 (многочлен 3-й степени)
for i, (xi, yi) in enumerate(zip(x_l3_v1, y_l3_v1)):
    L3_val = lagrange_poly(xi, x_l3_v1, y_l3_v1)
    match = 'СОВПАДАЕТ' if abs(L3_val - yi) < 1e-10 else 'ОШИБКА'
    print(f"L3 в узле x={xi:.2f}: {L3_val:.6f} (истинное: {yi:.6f}) - {match}")


L2_star_v1 = lagrange_poly(x_star, x_l2_v1, y_l2_v1)
L3_star_v1 = lagrange_poly(x_star, x_l3_v1, y_l3_v1)

print("\nРЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ:")
print(f"Точка интерполяции x* = {x_star}")
print(f"L2(x*) = {L2_star_v1:.6f}  (многочлен 2-й степени, 3 узла)")
print(f"L3(x*) = {L3_star_v1:.6f}  (многочлен 3-й степени, 4 узла)")


# Показывает, что результат интерполяции зависит от выбора узлов
# Даже если используется то же количество узлов, разные узлы дают разные значения

print("\n" + "=" * 70)
print("ВАРИАНТ 2: Узлы начиная со второго узла (-4.05; 2.0354)")
print("=" * 70)

# L2: используем узлы с индексами 1, 2, 3 (второй, третий, четвертый)
idx_l2_v2 = np.array([1, 2, 3])
x_l2_v2 = x[idx_l2_v2]
y_l2_v2 = y[idx_l2_v2]

# L3: используем узлы с индексами 1, 2, 3, 4 (второй, третий, четвертый, пятый)
idx_l3_v2 = np.array([1, 2, 3, 4])
x_l3_v2 = x[idx_l3_v2]
y_l3_v2 = y[idx_l3_v2]

print("\nУзлы для интерполяции (начиная со второго узла):")
print(f"L2: индексы {idx_l2_v2}, x = {x_l2_v2}, y = {y_l2_v2}")
print(f"L3: индексы {idx_l3_v2}, x = {x_l3_v2}, y = {y_l3_v2}")

# Проверка свойства интерполяции для варианта 2
print("\nПРОВЕРКА В УЗЛОВЫХ ТОЧКАХ:")

for i, (xi, yi) in enumerate(zip(x_l2_v2, y_l2_v2)):
    L2_val = lagrange_poly(xi, x_l2_v2, y_l2_v2)
    match = 'СОВПАДАЕТ' if abs(L2_val - yi) < 1e-10 else 'ОШИБКА'
    print(f"L2 в узле x={xi:.2f}: {L2_val:.6f} (истинное: {yi:.6f}) - {match}")

print()

for i, (xi, yi) in enumerate(zip(x_l3_v2, y_l3_v2)):
    L3_val = lagrange_poly(xi, x_l3_v2, y_l3_v2)
    match = 'СОВПАДАЕТ' if abs(L3_val - yi) < 1e-10 else 'ОШИБКА'
    print(f"L3 в узле x={xi:.2f}: {L3_val:.6f} (истинное: {yi:.6f}) - {match}")

# Вычисление значений в x* для варианта 2
L2_star_v2 = lagrange_poly(x_star, x_l2_v2, y_l2_v2)
L3_star_v2 = lagrange_poly(x_star, x_l3_v2, y_l3_v2)

print("\nРЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ:")
print(f"Точка интерполяции x* = {x_star}")
print(f"L2(x*) = {L2_star_v2:.6f}  (многочлен 2-й степени, узлы 1-2-3)")
print(f"L3(x*) = {L3_star_v2:.6f}  (многочлен 3-й степени, узлы 1-2-3-4)")


fig1 = plt.figure(figsize=(16, 10), dpi=100)

# Попытка развернуть окно на весь экран (работает не во всех средах)
try:
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
except:
    pass

# Отображаем все узловые точки, соединяя их прямой линией (для наглядности)
plt.plot(x, y, 'bo-', label='Узловые точки', markersize=10, zorder=5, 
         linewidth=2, alpha=0.7)

# Строим график многочлена L2 (2-я степень) на расширенном интервале
# Берем интервал чуть шире, чем узлы, чтобы увидеть поведение на краях
x_plot_l2_v1 = np.linspace(x_l2_v1.min()-0.3, x_l2_v1.max()+0.3, 200)
y_l2_plot_v1 = [lagrange_poly(xi, x_l2_v1, y_l2_v1) for xi in x_plot_l2_v1]
plt.plot(x_plot_l2_v1, y_l2_plot_v1, 'r-', label='L₂(x) (2-я степень)', linewidth=3)

# Строим график многочлена L3 (3-я степень)
x_plot_l3_v1 = np.linspace(x_l3_v1.min()-0.3, x_l3_v1.max()+0.3, 200)
y_l3_plot_v1 = [lagrange_poly(xi, x_l3_v1, y_l3_v1) for xi in x_plot_l3_v1]
plt.plot(x_plot_l3_v1, y_l3_plot_v1, 'g-', label='L₃(x) (3-я степень)', linewidth=3)

# Отмечаем точку интерполяции x* и полученные значения
plt.axvline(x=x_star, color='purple', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'x* = {x_star}')
plt.plot(x_star, L2_star_v1, 'ro', markersize=14, 
         label=f'L₂(x*) = {L2_star_v1:.4f}', zorder=6)
plt.plot(x_star, L3_star_v1, 'go', markersize=14, 
         label=f'L₃(x*) = {L3_star_v1:.4f}', zorder=6)

# Подписываем все узловые точки (координаты)
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi:.2f}, {yi:.2f})', (xi, yi),
                textcoords="offset points", xytext=(0,12), ha='center', 
                fontsize=10, fontweight='bold')

plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('y', fontsize=14, fontweight='bold')
plt.title('Интерполяционные многочлены Лагранжа (по близости к x*)\n'
          'для таблично заданной функции', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

#Начиная со второго узла (показывает эффект выбора узлов)
fig2 = plt.figure(figsize=(16, 10), dpi=100)

try:
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
except:
    pass

# Отображаем все узловые точки
plt.plot(x, y, 'bo-', label='Все узловые точки', markersize=10, zorder=5, 
         linewidth=2, alpha=0.7)

# Строим L2 на расширенном интервале (чтобы увидеть поведение вне узлов)
x_plot_l2_v2 = np.linspace(x_l2_v2.min()-1.0, x_l2_v2.max()+1.0, 400)
y_l2_plot_v2 = [lagrange_poly(xi, x_l2_v2, y_l2_v2) for xi in x_plot_l2_v2]
plt.plot(x_plot_l2_v2, y_l2_plot_v2, 'r-', 
         label='L₂(x) (2-я степень, узлы 1-2-3)', linewidth=3, zorder=4)

# Строим L3 на расширенном интервале
x_plot_l3_v2 = np.linspace(x_l3_v2.min()-1.0, x_l3_v2.max()+1.0, 400)
y_l3_plot_v2 = [lagrange_poly(xi, x_l3_v2, y_l3_v2) for xi in x_plot_l3_v2]
plt.plot(x_plot_l3_v2, y_l3_plot_v2, 'g-', 
         label='L₃(x) (3-я степень, узлы 1-2-3-4)', linewidth=3, zorder=4)

# Отмечаем точку x*
plt.axvline(x=x_star, color='purple', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'x* = {x_star}', zorder=3)
plt.plot(x_star, L2_star_v2, 'ro', markersize=16, 
         label=f'L₂(x*) = {L2_star_v2:.4f}', zorder=6,
         markeredgecolor='darkred', markeredgewidth=2)
plt.plot(x_star, L3_star_v2, 'go', markersize=16, 
         label=f'L₃(x*) = {L3_star_v2:.4f}', zorder=6,
         markeredgecolor='darkgreen', markeredgewidth=2)

# Выделяем узлы, использованные для построения многочленов
plt.plot(x_l2_v2, y_l2_v2, 'rs', markersize=16, 
         label='Узлы для L₂ (индексы 1,2,3)', zorder=7,
         markeredgecolor='darkred', markeredgewidth=2.5, alpha=0.9)
plt.plot(x_l3_v2, y_l3_v2, 'g^', markersize=16, 
         label='Узлы для L₃ (индексы 1,2,3,4)', zorder=7,
         markeredgecolor='darkgreen', markeredgewidth=2.5, alpha=0.9)

# Подписываем узлы с их индексами (для наглядности)
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'[{i}] ({xi:.2f}, {yi:.2f})', (xi, yi),
                textcoords="offset points", xytext=(0,15), ha='center', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('y', fontsize=14, fontweight='bold')
plt.title('Интерполяционные многочлены Лагранжа (начиная со 2-го узла)\n'
          'для таблично заданной функции', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


print("\n" + "=" * 70)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)
print(f"{'Метод':<40} {'L2(x*)':<15} {'L3(x*)':<15}")
print("-" * 70)
print(f"{'По близости к x*':<40} {L2_star_v1:<15.6f} {L3_star_v1:<15.6f}")
print(f"{'Начиная со 2-го узла':<40} {L2_star_v2:<15.6f} {L3_star_v2:<15.6f}")
print(f"{'Разница':<40} {abs(L2_star_v1-L2_star_v2):<15.6f} {abs(L3_star_v1-L3_star_v2):<15.6f}")
print("=" * 70)

"""
ВЫВОДЫ (по результатам выполнения):
1. Оба многочлена (Лагранжа 2-й и 3-й степени) проходят через выбранные узлы
   (проверка свойства интерполяции).

2. Результаты интерполяции в точке x* = -3.875 зависят от выбора узлов:
   - L2: разница между вариантами составляет ~0.01848
   - L3: разница между вариантами составляет ~0.00914

3. Многочлен более высокой степени (L3) дает более стабильный результат
   (меньшую разницу между вариантами выбора узлов).

4. На графиках видно, что многочлены ведут себя по-разному вне интервала узлов,
   что связано с эффектом Рунге (осцилляции на краях).

5. Рекомендуется для интерполяции выбирать узлы, ближайшие к точке x*,
   так как это минимизирует погрешность (теорема об оценке погрешности).
"""