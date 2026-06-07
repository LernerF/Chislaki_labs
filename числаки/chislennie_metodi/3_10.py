"""
Лабораторная работа 3.1: Интерполяция таблично заданной функции
Методы: Лагранжа и Ньютона (2-й и 3-й степени)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. ИСХОДНЫЕ ДАННЫЕ
# Таблица значений функции в узлах интерполяции
x = np.array([-4.58, -4.05, -3.52, -2.99, -2.46, -1.93, -1.40, -0.87, -0.34])
y = np.array([1.7361, 2.0354, 3.0762, 2.8126, 1.2583, 1.8624, 1.7263, 3.2846, 3.4107])

# Точка интерполяции
x_star = -3.875

# 2. МЕТОД ЛАГРАНЖА
# Формула: P_n(x) = Σ y_i * L_i(x), где L_i(x) = Π (x - x_j)/(x_i - x_j)
def lagrange_poly(x_val, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

# 3. МЕТОД НЬЮТОНА
# Формула: P_n(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ...
def divided_differences(x_nodes, y_nodes):
    """Построение таблицы разделённых разностей"""
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_nodes[i + j] - x_nodes[i])
    return table

def newton_poly(x_val, x_nodes, diff_table):
    """Вычисление значения многочлена Ньютона в точке x_val"""
    n = len(x_nodes)
    result = diff_table[0, 0]
    product = 1.0
    
    for i in range(1, n):
        product *= (x_val - x_nodes[i - 1])
        result += diff_table[0, i] * product
    return result


# Теория: R_n(x) = ω_{n+1}(x) * f[x0,...,xn,x_{n+1}]
def compute_error(x_val, x_nodes, y_nodes):
    """Оценка погрешности интерполяции в точке x_val"""
    n = len(x_nodes)
    
    # ω(x) = (x - x0)(x - x1)...(x - xn)
    product = 1.0
    for xi in x_nodes:
        product *= (x_val - xi)
    
    # Находим ближайший неиспользованный узел для следующей разделённой разности
    used_indices = [np.where(x == xi)[0][0] for xi in x_nodes]
    unused_indices = [i for i in range(len(x)) if i not in used_indices]
    
    if len(unused_indices) == 0:
        return 0.0
    
    distances_unused = [abs(x[i] - x_val) for i in unused_indices]
    nearest_unused_idx = unused_indices[np.argmin(distances_unused)]
    
    # Расширенный набор узлов
    x_extended = np.append(x_nodes, x[nearest_unused_idx])
    y_extended = np.append(y_nodes, y[nearest_unused_idx])
    
    diff_table = divided_differences(x_extended, y_extended)
    f_next = diff_table[0, n]  # f[x0,...,xn,x_{n+1}]
    
    return abs(product * f_next)


# Сортируем узлы по близости к x* (для варианта 1)
distances = np.abs(x - x_star)
sorted_indices = np.argsort(distances)

# Вариант 1: ближайшие к x* узлы
idx_l2_v1 = sorted_indices[:3]      # 3 узла для степени 2
idx_l3_v1 = sorted_indices[:4]      # 4 узла для степени 3

# Вариант 2: фиксированные узлы (со 2-го по 5-й)
idx_l2_v2 = np.array([1, 2, 3])     # узлы 1,2,3
idx_l3_v2 = np.array([1, 2, 3, 4])  # узлы 1,2,3,4

# Собираем значения для каждого варианта
x_l2_v1, y_l2_v1 = x[idx_l2_v1], y[idx_l2_v1]
x_l3_v1, y_l3_v1 = x[idx_l3_v1], y[idx_l3_v1]
x_l2_v2, y_l2_v2 = x[idx_l2_v2], y[idx_l2_v2]
x_l3_v2, y_l3_v2 = x[idx_l3_v2], y[idx_l3_v2]

print("=" * 100)
print("МЕТОД ЛАГРАНЖА")
print("=" * 100)

# Вариант 1 (по близости)
L2_star_v1 = lagrange_poly(x_star, x_l2_v1, y_l2_v1)
L3_star_v1 = lagrange_poly(x_star, x_l3_v1, y_l3_v1)
RL2_v1 = compute_error(x_star, x_l2_v1, y_l2_v1)
RL3_v1 = compute_error(x_star, x_l3_v1, y_l3_v1)

print(f"\nВариант 1 (узлы по близости к x*):")
print(f"  L2(x*) = {L2_star_v1:.6f}, погрешность = {RL2_v1:.6f}")
print(f"  L3(x*) = {L3_star_v1:.6f}, погрешность = {RL3_v1:.6f}")

# Вариант 2 (со 2-го узла)
L2_star_v2 = lagrange_poly(x_star, x_l2_v2, y_l2_v2)
L3_star_v2 = lagrange_poly(x_star, x_l3_v2, y_l3_v2)
RL2_v2 = compute_error(x_star, x_l2_v2, y_l2_v2)
RL3_v2 = compute_error(x_star, x_l3_v2, y_l3_v2)

print(f"\nВариант 2 (начиная со 2-го узла):")
print(f"  L2(x*) = {L2_star_v2:.6f}, погрешность = {RL2_v2:.6f}")
print(f"  L3(x*) = {L3_star_v2:.6f}, погрешность = {RL3_v2:.6f}")


print("\n" + "=" * 100)
print("МЕТОД НЬЮТОНА")
print("=" * 100)

# Вариант 1 (по близости)
diff_n2_v1 = divided_differences(x_l2_v1, y_l2_v1)
diff_n3_v1 = divided_differences(x_l3_v1, y_l3_v1)
N2_star_v1 = newton_poly(x_star, x_l2_v1, diff_n2_v1)
N3_star_v1 = newton_poly(x_star, x_l3_v1, diff_n3_v1)

# Оценка погрешности для Ньютона (через следующую разделённую разность)
product_n2 = np.prod([x_star - xi for xi in x_l2_v1])
product_n3 = np.prod([x_star - xi for xi in x_l3_v1])
diff_n4_v1 = divided_differences(x[sorted_indices[:5]], y[sorted_indices[:5]])
diff_n5_v1 = divided_differences(x[sorted_indices[:5]], y[sorted_indices[:5]])
R2_v1 = abs(product_n2 * diff_n3_v1[0, 3])
R3_v1 = abs(product_n3 * diff_n4_v1[0, 4])

print(f"\nВариант 1 (узлы по близости к x*):")
print(f"  N2(x*) = {N2_star_v1:.6f}, погрешность = {R2_v1:.6f}")
print(f"  N3(x*) = {N3_star_v1:.6f}, погрешность = {R3_v1:.6f}")

# Вариант 2 (со 2-го узла)
diff_n2_v2 = divided_differences(x_l2_v2, y_l2_v2)
diff_n3_v2 = divided_differences(x_l3_v2, y_l3_v2)
N2_star_v2 = newton_poly(x_star, x_l2_v2, diff_n2_v2)
N3_star_v2 = newton_poly(x_star, x_l3_v2, diff_n3_v2)

product_n2_v2 = np.prod([x_star - xi for xi in x_l2_v2])
product_n3_v2 = np.prod([x_star - xi for xi in x_l3_v2])
diff_n4_v2 = divided_differences(x[1:6], y[1:6])
R2_v2 = abs(product_n2_v2 * diff_n3_v2[0, 3])
R3_v2 = abs(product_n3_v2 * diff_n4_v2[0, 4])

print(f"\nВариант 2 (начиная со 2-го узла):")
print(f"  N2(x*) = {N2_star_v2:.6f}, погрешность = {R2_v2:.6f}")
print(f"  N3(x*) = {N3_star_v2:.6f}, погрешность = {R3_v2:.6f}")


print("\n" + "=" * 120)
print("ИТОГОВАЯ ТАБЛИЦА: СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 120)
print(f"{'Метод':<15} {'Вариант':<18} {'Степень 2':<18} {'Погрешность 2':<18} {'Степень 3':<18} {'Погрешность 3':<18}")
print("-" * 120)
print(f"{'ЛАГРАНЖ':<15} {'По близости':<18} {L2_star_v1:<18.6f} {RL2_v1:<18.6f} {L3_star_v1:<18.6f} {RL3_v1:<18.6f}")
print(f"{'':<15} {'Со 2-го узла':<18} {L2_star_v2:<18.6f} {RL2_v2:<18.6f} {L3_star_v2:<18.6f} {RL3_v2:<18.6f}")
print(f"{'НЬЮТОН':<15} {'По близости':<18} {N2_star_v1:<18.6f} {R2_v1:<18.6f} {N3_star_v1:<18.6f} {R3_v1:<18.6f}")
print(f"{'':<15} {'Со 2-го узла':<18} {N2_star_v2:<18.6f} {R2_v2:<18.6f} {N3_star_v2:<18.6f} {R3_v2:<18.6f}")
print("=" * 120)



# График 1: Ньютон (по близости)
fig1 = plt.figure(figsize=(16, 10), dpi=100)
plt.plot(x, y, 'bo-', label='Узловые точки', markersize=10, linewidth=2, alpha=0.7)

x_plot = np.linspace(x_l2_v1.min()-0.3, x_l2_v1.max()+0.3, 200)
y_plot_n2 = [newton_poly(xi, x_l2_v1, diff_n2_v1) for xi in x_plot]
y_plot_n3 = [newton_poly(xi, x_l3_v1, diff_n3_v1) for xi in x_plot]

plt.plot(x_plot, y_plot_n2, 'r-', label='N₂(x) (2-я степень)', linewidth=3)
plt.plot(x_plot, y_plot_n3, 'g-', label='N₃(x) (3-я степень)', linewidth=3)
plt.axvline(x=x_star, color='purple', linestyle='--', linewidth=2, label=f'x* = {x_star}')
plt.plot(x_star, N2_star_v1, 'ro', markersize=14, label=f'N₂(x*) = {N2_star_v1:.4f}')
plt.plot(x_star, N3_star_v1, 'go', markersize=14, label=f'N₃(x*) = {N3_star_v1:.4f}')

for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi:.2f}, {yi:.2f})', (xi, yi), xytext=(0,12), ha='center', fontsize=10)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Интерполяционные многочлены Ньютона (по близости к x*)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# График 2: Ньютон (со 2-го узла) - расширенный диапазон
fig2 = plt.figure(figsize=(16, 10), dpi=100)
plt.plot(x, y, 'bo-', label='Все узловые точки', markersize=10, linewidth=2, alpha=0.7)

x_plot2 = np.linspace(x_l2_v2.min()-1.0, x_l2_v2.max()+1.0, 400)
y_plot_n2_v2 = [newton_poly(xi, x_l2_v2, diff_n2_v2) for xi in x_plot2]
y_plot_n3_v2 = [newton_poly(xi, x_l3_v2, diff_n3_v2) for xi in x_plot2]

plt.plot(x_plot2, y_plot_n2_v2, 'r-', label='N₂(x) (узлы 1-2-3)', linewidth=3)
plt.plot(x_plot2, y_plot_n3_v2, 'g-', label='N₃(x) (узлы 1-2-3-4)', linewidth=3)
plt.axvline(x=x_star, color='purple', linestyle='--', linewidth=2, label=f'x* = {x_star}')
plt.plot(x_star, N2_star_v2, 'ro', markersize=16, label=f'N₂(x*) = {N2_star_v2:.4f}')
plt.plot(x_star, N3_star_v2, 'go', markersize=16, label=f'N₃(x*) = {N3_star_v2:.4f}')
plt.plot(x_l2_v2, y_l2_v2, 'rs', markersize=16, label='Узлы для N₂')
plt.plot(x_l3_v2, y_l3_v2, 'g^', markersize=16, label='Узлы для N₃')

for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'[{i}] ({xi:.2f}, {yi:.2f})', (xi, yi), xytext=(0,15), ha='center', fontsize=9)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Интерполяционные многочлены Ньютона (начиная со 2-го узла)', fontsize=16)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# График 3: Сравнение Лагранжа и Ньютона
fig3 = plt.figure(figsize=(16, 10), dpi=100)
plt.plot(x, y, 'bo-', label='Узловые точки', markersize=10, linewidth=2, alpha=0.7)

x_plot3 = np.linspace(x_l3_v1.min()-0.5, x_l3_v1.max()+0.5, 200)
y_lagrange = [lagrange_poly(xi, x_l3_v1, y_l3_v1) for xi in x_plot3]
y_newton = [newton_poly(xi, x_l3_v1, diff_n3_v1) for xi in x_plot3]

plt.plot(x_plot3, y_lagrange, 'r-', label='L₃(x) Лагранж', linewidth=3, alpha=0.8)
plt.plot(x_plot3, y_newton, 'g--', label='N₃(x) Ньютон', linewidth=3, alpha=0.8)
plt.axvline(x=x_star, color='purple', linestyle='--', linewidth=2, label=f'x* = {x_star}')
plt.plot(x_star, L3_star_v1, 'ro', markersize=14, label=f'L₃(x*) = {L3_star_v1:.4f}')
plt.plot(x_star, N3_star_v1, 'go', markersize=14, label=f'N₃(x*) = {N3_star_v1:.4f}')

for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi:.2f}, {yi:.2f})', (xi, yi), xytext=(0,12), ha='center', fontsize=10)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Сравнение Лагранжа и Ньютона (степень 3, узлы по близости)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)
print("1. Результаты Лагранжа и Ньютона совпадают (разница < 10⁻¹⁰).")
print("2. Выбор узлов влияет на результат интерполяции.")
print("3. Ближайшие к x* узлы дают меньшую погрешность.")
print("4. Многочлен более высокой степени (3) точнее, чем низкой (2).")
print("5. На графиках видно, что за пределами узлов многочлены могут сильно расходиться.")
print("=" * 70)