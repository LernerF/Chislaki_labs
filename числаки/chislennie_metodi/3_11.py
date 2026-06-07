import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка отображения графиков
rcParams['figure.figsize'] = (12, 10)
rcParams['font.size'] = 10

print("=" * 80)
print("ПОСТРОЕНИЕ ЕСТЕСТВЕННОГО КУБИЧЕСКОГО СПЛАЙНА ДЕФЕКТА 1")
print("=" * 80)

i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = np.array([-3.25, -2.706, -2.230, -1.822, -1.142, -0.394, 0.15, 0.762, 2.054, 2.938, 3.55])
y = np.array([5.704, 3.841, 0.273, -1.817, -0.354, 1.986, 0.185, -1.861, -0.403, -3.274, -5.899])
x_star = -1.905  # точка, в которой нужно найти значение

print(f"n = {len(x) - 1} интервалов (всего узлов сетки: {len(x)})")
print(f"\nТаблица исходных данных:")
print(f"{'i':<3} {'x_i':<12} {'y_i':<12}")
print("-" * 28)
for j in range(len(x)):
    print(f"{j:<3} {x[j]:<12.3f} {y[j]:<12.3f}")
print(f"\nТочка интерполяции: x* = {x_star}")

n = len(x) - 1  # количество интервалов
h = np.diff(x)  # шаги между узлами: h_i = x_{i+1} - x_i

# сетка
print("\n" + "=" * 80)
print("ШАГ 1: ВЫЧИСЛЕНИЕ ШАГОВ СЕТКИ")
print("=" * 80)

print(f"\nШаги сетки:")
for i_idx in range(1, n + 1):
    print(f"h_{i_idx} = {x[i_idx]:.3f} - {x[i_idx - 1]:.3f} = {h[i_idx - 1]:.4f}")

# уравнения
# Решаем систему для нахождения c_i (вторых производных в узлах)
print("\n" + "=" * 80)
print("ШАГ 2: ПОСТРОЕНИЕ СИСТЕМЫ УРАВНЕНИЙ")
print("=" * 80)

A = np.zeros((n + 1, n + 1))  # матрица системы
b_vec = np.zeros(n + 1)        # вектор правой части

# Граничное условие: c_0 = 0 (нулевая кривизна на левом конце)
A[0, 0] = 1
b_vec[0] = 0

# Заполняем уравнения для внутренних узлов (i = 1,..., n-1)
for i_idx in range(1, n):
    coeff_left = h[i_idx - 1]                     # коэффициент при c_{i-1}
    coeff_center = 2 * (h[i_idx - 1] + h[i_idx]) # коэффициент при c_i
    coeff_right = h[i_idx]                       # коэффициент при c_{i+1}
    
    # Правая часть: 3 * (Δy_i/h_i - Δy_{i-1}/h_{i-1})
    rhs = 3 * ((y[i_idx + 1] - y[i_idx]) / h[i_idx] -
               (y[i_idx] - y[i_idx - 1]) / h[i_idx - 1])
    
    A[i_idx, i_idx - 1] = coeff_left
    A[i_idx, i_idx] = coeff_center
    A[i_idx, i_idx + 1] = coeff_right
    b_vec[i_idx] = rhs

# Граничное условие: c_n = 0 (нулевая кривизна на правом конце)
A[n, n] = 1
b_vec[n] = 0

#решение
print("\n" + "=" * 80)
print("ШАГ 3: РЕШЕНИЕ СИСТЕМЫ УРАВНЕНИЙ")
print("=" * 80)

c = np.linalg.solve(A, b_vec)  # находим c_i

print(f"\nКоэффициенты c_i:")
for i_idx in range(n + 1):
    print(f"c_{i_idx} = {c[i_idx]:10.6f}", end="")
    if (i_idx + 1) % 3 == 0:
        print()
    else:
        print("  |  ", end="")

# Находим a_i, b_i, d_i для каждого интервала
print("\n\n" + "=" * 80)
print("ШАГ 4: ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ a_i, b_i, d_i")
print("=" * 80)

a = np.zeros(n)      # a_i = y_i (значение в левом конце)
b_coef = np.zeros(n) # коэффициент при (x-x_i)
d = np.zeros(n)      # коэффициент при (x-x_i)³

for i_idx in range(n):
    a[i_idx] = y[i_idx]  # значение функции в левом узле интервала

# Формула для b_i
for i_idx in range(n):
    b_coef[i_idx] = (y[i_idx + 1] - y[i_idx]) / h[i_idx] - (h[i_idx] / 3) * (2 * c[i_idx] + c[i_idx + 1])

# Формула для d_i
for i_idx in range(n):
    d[i_idx] = (c[i_idx + 1] - c[i_idx]) / (3 * h[i_idx])

print(f"\nВычисленные коэффициенты для всех интервалов:")
print(f"{'Инт':<4} {'a_i':<12} {'b_i':<12} {'c_i':<12} {'d_i':<12}")
print("-" * 49)
for i_idx in range(n):
    print(f"{i_idx:<4} {a[i_idx]:<12.6f} {b_coef[i_idx]:<12.6f} {c[i_idx]:<12.6f} {d[i_idx]:<12.6f}")

# поиск интервала
# Находим, на каком интервале находится точка x_star
print("\n" + "=" * 80)
print("ШАГ 5: НАХОЖДЕНИЕ ИНТЕРВАЛА ДЛЯ ТОЧКИ x*")
print("=" * 80)

k = np.searchsorted(x, x_star) - 1  # бинарный поиск интервала
if k < 0:
    k = 0
elif k >= n:
    k = n - 1

print(f"\nТочка x* = {x_star} находится на интервале [{x[k]:.3f}, {x[k + 1]:.3f}]")
print(f"Интервал номер: i = {k} (отсчет от 0)")
print(f"Интервал номер: i = {k + 1} (отсчет от 1)")

print("\n" + "=" * 80)
print("ШАГ 6: КОЭФФИЦИЕНТЫ СПЛАЙНА В ИНТЕРВАЛЕ, СОДЕРЖАЩЕМ x*")
print("=" * 80)

print(f"\nДля интервала [{x[k]:.3f}, {x[k + 1]:.3f}] (i = {k}):")
print("-" * 60)
print(f"Коэффициент a_{k} = y_{k} = {a[k]:.6f}")
print(f"Коэффициент b_{k} = {b_coef[k]:.6f}")
print(f"Коэффициент c_{k} = {c[k]:.6f}")
print(f"Коэффициент d_{k} = {d[k]:.6f}")
print("-" * 60)
print(f"Функция сплайна на этом интервале:")
print(
    f"S_{k}(x) = {a[k]:.6f} + {b_coef[k]:.6f}·(x - {x[k]:.3f}) + {c[k]:.6f}·(x - {x[k]:.3f})² + {d[k]:.6f}·(x - {x[k]:.3f})³")

# ВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ 
print("\n" + "=" * 80)
print("ШАГ 7: ВЫЧИСЛЕНИЕ ЗНАЧЕНИЯ СПЛАЙНА И ЕГО ПРОИЗВОДНЫХ")
print("=" * 80)

dx = x_star - x[k]  # расстояние от левого узла до точки x*
s3_value = a[k] + b_coef[k] * dx + c[k] * dx ** 2 + d[k] * dx ** 3           # значение функции
s3_prime = b_coef[k] + 2 * c[k] * dx + 3 * d[k] * dx ** 2                     # первая производная
s3_double_prime = 2 * c[k] + 6 * d[k] * dx                                    # вторая производная

print(f"\nВычисления в точке x* = {x_star}:")
print(f"dx = x* - x_{k} = {x_star} - {x[k]} = {dx:.6f}")
print(
    f"\nS_3({x_star})    = {a[k]:.6f} + {b_coef[k]:.6f}·{dx:.6f} + {c[k]:.6f}·{dx:.6f}² + {d[k]:.6f}·{dx:.6f}³ = {s3_value:.6f}")
print(f"S_3'({x_star})   = {b_coef[k]:.6f} + 2·{c[k]:.6f}·{dx:.6f} + 3·{d[k]:.6f}·{dx:.6f}² = {s3_prime:.6f}")
print(f"S_3''({x_star})  = 2·{c[k]:.6f} + 6·{d[k]:.6f}·{dx:.6f} = {s3_double_prime:.6f}")

def spline_value(x_val, interval_idx):
    """Вычисляет значение сплайна в точке x_val на заданном интервале"""
    if interval_idx < 0 or interval_idx >= n:
        return None
    dx = x_val - x[interval_idx]
    return a[interval_idx] + b_coef[interval_idx] * dx + c[interval_idx] * dx ** 2 + d[interval_idx] * dx ** 3

def spline_prime(x_val, interval_idx):
    """Вычисляет первую производную сплайна"""
    if interval_idx < 0 or interval_idx >= n:
        return None
    dx = x_val - x[interval_idx]
    return b_coef[interval_idx] + 2 * c[interval_idx] * dx + 3 * d[interval_idx] * dx ** 2

def spline_double_prime(x_val, interval_idx):
    """Вычисляет вторую производную сплайна"""
    if interval_idx < 0 or interval_idx >= n:
        return None
    dx = x_val - x[interval_idx]
    return 2 * c[interval_idx] + 6 * d[interval_idx] * dx

def find_interval(xs, x_val):
    """Находит индекс интервала, содержащего x_val"""
    n_points = len(xs)
    if x_val < xs[0] - 1e-9 or x_val > xs[n_points - 1] + 1e-9:
        return -1
    if abs(x_val - xs[n_points - 1]) < 1e-9:
        return n_points - 2
    return np.searchsorted(xs, x_val, side='right') - 1

# Создаем точки для плавного отображения графиков
x_plot = np.linspace(x[0], x[-1], 1000)
y_plot = np.array([spline_value(xi, find_interval(x, xi)) for xi in x_plot])           # значения сплайна
y1_plot = np.array([spline_prime(xi, find_interval(x, xi)) for xi in x_plot])          # первая производная
y2_plot = np.array([spline_double_prime(xi, find_interval(x, xi)) for xi in x_plot])   # вторая производная

# Создаем окно с 4 графиками: 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1.2]})
ax1 = axes[0, 0]      # основной график сплайна
ax_fill = axes[0, 1]  # график с заливкой по интервалам
ax_d1 = axes[1, 0]    # первая производная
ax_d2 = axes[1, 1]    # вторая производная

# ГРАФИК 1: Основной сплайн
ax1.plot(x_plot, y_plot, linewidth=1.5, label='S(x)', color='#1f77b4')
ax1.scatter(x, y, zorder=5, label='Узлы интерполяции', color='red')
ax1.scatter([x_star], [s3_value], s=50, marker='o', zorder=10,
            color='green', label=f'S({x_star}) = {s3_value:.6f}')
ax1.annotate(f'S({x_star}) = {s3_value:.6f}', xy=(x_star, s3_value),
             xytext=(10, 10), textcoords='offset points', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax1.set_ylabel('S(x)')
ax1.set_title('Естественный кубический сплайн')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

cmap = plt.get_cmap('tab20')
baseline = np.min(y_plot) - 0.1 * (np.max(y_plot) - np.min(y_plot))

for i in range(n):
    x_segment = np.linspace(x[i], x[i + 1], 50)
    y_segment = np.array([spline_value(xi, i) for xi in x_segment])
    color = cmap(i % cmap.N)
    
    # Выделяем интервал, содержащий x* (делаем его ярче)
    alpha = 0.7 if i == k else 0.3
    linewidth = 2.0 if i == k else 0.9
    
    ax_fill.fill_between(x_segment, y_segment, y2=baseline, facecolor=color,
                         alpha=alpha, edgecolor='none')
    ax_fill.plot(x_segment, y_segment, linewidth=linewidth, color=color,
                 label=f'Интервал {i}' if i == k else None)

ax_fill.scatter(x, y, zorder=5, color='red')
ax_fill.scatter([x_star], [s3_value], s=50, marker='o', zorder=10,
                color='green', label=f'S({x_star}) = {s3_value:.6f}')
ax_fill.set_ylabel('S(x) - заливка сегментов')
ax_fill.set_title(f'Сегменты сплайна (выделен интервал {k}, содержащий x*)')
ax_fill.grid(True, alpha=0.3)
ax_fill.set_xlim(x[0], x[-1])
ax_fill.legend(loc='best')

# ГРАФИК 3: Первая производная (скорость изменения)
ax_d1.plot(x_plot, y1_plot, linewidth=1.3, label="S'(x)", color='#ff7f0e')
ax_d1.scatter([x_star], [s3_prime], s=40, marker='o', zorder=10,
              color='green', label=f"S'({x_star}) = {s3_prime:.6f}")
ax_d1.annotate(f"S'({x_star}) = {s3_prime:.6f}", xy=(x_star, s3_prime),
               xytext=(10, 10), textcoords='offset points', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_d1.set_ylabel("S'(x)")
ax_d1.set_xlabel('x')
ax_d1.grid(True, alpha=0.3)
ax_d1.legend(loc='best')

# ГРАФИК 4: Вторая производная (кривизна)
ax_d2.plot(x_plot, y2_plot, linewidth=1.2, label="S''(x)", color='#2ca02c')
ax_d2.axhline(0, linestyle='--', linewidth=0.8, color='gray', alpha=0.5)  # нулевая линия
ax_d2.scatter([x_star], [s3_double_prime], s=40, marker='o', zorder=10,
              color='green', label=f"S''({x_star}) = {s3_double_prime:.6f}")
ax_d2.annotate(f"S''({x_star}) = {s3_double_prime:.6f}", xy=(x_star, s3_double_prime),
               xytext=(10, 10), textcoords='offset points', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_d2.scatter(x[0], 0, s=30, color='blue', zorder=5)   # отмечаем нулевую кривизну на левом конце
ax_d2.scatter(x[-1], 0, s=30, color='blue', zorder=5)  # отмечаем нулевую кривизну на правом конце
ax_d2.set_ylabel("S''(x)")
ax_d2.set_xlabel('x')
ax_d2.grid(True, alpha=0.3)
ax_d2.legend(loc='best')

plt.tight_layout()
plt.savefig('cubic_spline_variant_17_new.png', dpi=200, bbox_inches='tight')
plt.show()

# ==================== ВЫВОД ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ ====================
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 80)

print(f"\n>>> ТОЧКА ИНТЕРПОЛЯЦИИ: x* = {x_star}")
print(f">>> ИНТЕРВАЛ: [{x[k]:.3f}, {x[k + 1]:.3f}] (i = {k})")

print(f"\n>>> КОЭФФИЦИЕНТЫ СПЛАЙНА В ЭТОМ ИНТЕРВАЛЕ:")
print(f"    a_{k} = {a[k]:.6f}")
print(f"    b_{k} = {b_coef[k]:.6f}")
print(f"    c_{k} = {c[k]:.6f}")
print(f"    d_{k} = {d[k]:.6f}")

print(f"\n>>> ЗНАЧЕНИЯ В ТОЧКЕ x* = {x_star}:")
print(f"    S_3({x_star})    = {s3_value:.6f}")
print(f"    S_3'({x_star})   = {s3_prime:.6f}")
print(f"    S_3''({x_star})  = {s3_double_prime:.6f}")

print("\n>>> ФОРМУЛА СПЛАЙНА В ЭТОМ ИНТЕРВАЛЕ:")
print(
    f"    S_{k}(x) = {a[k]:.6f} + {b_coef[k]:.6f}·(x - {x[k]:.3f}) + {c[k]:.6f}·(x - {x[k]:.3f})² + {d[k]:.6f}·(x - {x[k]:.3f})³")