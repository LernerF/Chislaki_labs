import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")


def exact_solution(x_list):
    """Аналитическое решение: y = x^2 - x^2/(ln(2x))"""
    return [x ** 2 - x ** 2 / np.log(2 * x) for x in x_list]


def f(x, y):
    """Правая часть уравнения: y' = (x^4 + y^2) / x^3"""
    if x == 0:
        return 0
    return (x ** 4 + y ** 2) / x ** 3


x0 = 1.0
y0 = 1.0 - 1.0 / np.log(2)  # y(1) = 1 - 1/ln(2)
x_end = 4.0
h = 0.20

# Создание сетки узлов
x = []
val = x0
while val < x_end + 0.5 * h:
    x.append(val)
    val += h

n = len(x)


def prod(b, K):
    """Вычисляет скалярное произведение коэффициентов b и стадий K"""
    return sum(b[j] * K[j] for j in range(len(b)))


# ===== ЯВНЫЕ МЕТОДЫ РУНГЕ-КУТТЫ =====
def explicit_runge_kutta(butcher_table, name=""):
    A, b, c = butcher_table
    p = len(b)

    y = [0.0 for _ in range(n)]
    y[0] = y0

    for i in range(n - 1):
        K = [0.0 for _ in range(p)]

        for j in range(p):
            sum_AK = 0
            for k in range(j):
                sum_AK += A[j][k] * K[k]

            K[j] = h * f(x[i] + c[j] * h, y[i] + sum_AK)

        y[i + 1] = y[i] + prod(b, K)

    return y


# ===== ТАБЛИЦЫ БУТЧЕРА =====
# Метод Эйлера (1 порядок)
euler_table = (
    [[0]],
    [1],
    [0]
)

# Метод Хойна 2-го порядка (2 порядок)
heun2_table = (
    [[0, 0],
     [1 / 2, 0]],
    [0, 1],
    [0, 1 / 2]
)

# Метод Хойна 3-го порядка (3 порядок)
heun3_table = (
    [[0, 0, 0],
     [1 / 3, 0, 0],
     [0, 2 / 3, 0]],
    [1 / 4, 0, 3 / 4],
    [0, 1 / 3, 2 / 3]
)

# Классический метод Рунге-Кутты 4-го порядка (4 порядок)
rk4_table = (
    [[0, 0, 0, 0],
     [1 / 2, 0, 0, 0],
     [0, 1 / 2, 0, 0],
     [0, 0, 1, 0]],
    [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    [0, 1 / 2, 1 / 2, 1]
)


# ===== ВЛОЖЕННЫЙ МЕТОД МЕРСОНА =====
def embedded_runge_kutta(butcher_table, name=""):
    A, b, b_star, c = butcher_table
    p = len(b)

    y = [0.0 for _ in range(n)]
    y[0] = y0

    for i in range(n - 1):
        K = [0.0 for _ in range(p)]

        for j in range(p):
            sum_AK = 0
            for k in range(j):
                sum_AK += A[j][k] * K[k]

            K[j] = h * f(x[i] + c[j] * h, y[i] + sum_AK)

        y[i + 1] = y[i] + prod(b, K)

    return y


# Таблица Бутчера для метода Мерсона 3(4)
merson_table = (
    [[0, 0, 0, 0, 0],
     [1 / 3, 0, 0, 0, 0],
     [1 / 6, 1 / 6, 0, 0, 0],
     [1 / 8, 0, 3 / 8, 0, 0],
     [1 / 2, 0, -3 / 2, 2, 0]],
    [1 / 10, 0, 3 / 10, 2 / 5, 1 / 5],
    [1 / 6, 0, 0, 2 / 3, 1 / 6],
    [0, 1 / 3, 1 / 3, 1 / 2, 1]
)


# ===== НЕЯВНЫЕ МЕТОДЫ РУНГЕ-КУТТЫ =====
def fixed_point_K(A, b, c, f, x, y, h, s, i, K, tol=1e-10, max_iter=100):
    for iteration in range(max_iter):
        K_new = []

        for j in range(s):
            sum_AK = 0
            for k in range(s):
                sum_AK += A[j][k] * K[k]

            val = h * f(x[i] + c[j] * h, y[i] + sum_AK)
            K_new.append(val)

        # Проверка сходимости
        if max(abs(K_new[j] - K[j]) for j in range(s)) < tol:
            return K_new

        K = K_new

    raise Exception("Стадии K не сошлись за max_iter итераций")


def implicit_runge_kutta(butcher_table, name=""):
    A, b, c = butcher_table
    s = len(b)

    y = [0.0 for _ in range(n)]
    y[0] = y0

    for i in range(n - 1):
        # Начальное приближение K (явный шаг Эйлера)
        K0 = [h * f(x[i] + c[j] * h, y[i]) for j in range(s)]

        # Решаем нелинейную систему
        K_sol = fixed_point_K(A, b, c, f, x, y, h, s, i, K0)

        y[i + 1] = y[i] + prod(b, K_sol)

    return y


# Таблица Бутчера для метода Гаусса 2-го порядка
gauss2_table = (
    [[1 / 2]],
    [1],
    [1 / 2]
)

# Таблица Бутчера для метода Гаусса 4-го порядка
sqrt3 = 3 ** 0.5
gauss4_table = (
    [[1 / 4, 1 / 4 - sqrt3 / 6],
     [1 / 4 + sqrt3 / 6, 1 / 4]],
    [1 / 2, 1 / 2],
    [1 / 2 - sqrt3 / 6, 1 / 2 + sqrt3 / 6]
)


# ===== МЕТОДЫ АДАМСА =====
def adams_moulton_simple_iter(i, x, y, h, coefficients, f, tol=1e-6, max_iter=100):
    y_next = y[i]

    for _ in range(max_iter):
        sum_coeff = 0

        for j in range(len(coefficients)):
            if j == 0:
                sum_coeff += coefficients[j] * f(x[i + 1], y_next)
            else:
                sum_coeff += coefficients[j] * f(x[i + 1 - j], y[i + 1 - j])

        y_new = y[i] + h * sum_coeff

        if abs(y_new - y_next) < tol:
            return y_new

        y_next = y_new

    raise Exception("Не сошлось")


def adams_method(coefficients, implicit=False, name=""):
    k = len(coefficients) - 1

    y = [0.0 for _ in range(n)]
    y_rk4 = explicit_runge_kutta(rk4_table)

    y[:k + 1] = y_rk4[:k + 1]

    for i in range(k, n - 1):
        if implicit:
            # Неявный метод Адамса-Моултона
            y[i + 1] = adams_moulton_simple_iter(i, x, y, h, coefficients, f)
        else:
            # Явный метод Адамса-Башфорта
            sum_coeff = 0
            for j in range(k + 1):
                sum_coeff += coefficients[j] * f(x[i - j], y[i - j])

            y[i + 1] = y[i] + h * sum_coeff

    return y


# Коэффициенты методов Адамса
adams_bashforth_coeff = [55 / 24, -59 / 24, 37 / 24, -9 / 24]
adams_moulton_coeff = [9 / 24, 19 / 24, -5 / 24, 1 / 24]

# ===== ВЫЧИСЛЕНИЕ РЕШЕНИЙ ВСЕМИ МЕТОДАМИ =====
methods = {
    'Явный метод Эйлера': explicit_runge_kutta(euler_table),
    'Метод Хойна 2 порядка': explicit_runge_kutta(heun2_table),
    'Метод Хойна 3 порядка': explicit_runge_kutta(heun3_table),
    'Классический RK4': explicit_runge_kutta(rk4_table),
    'Мерсон 3(4)': embedded_runge_kutta(merson_table),
    'Неявный метод Гаусса 2 порядка': implicit_runge_kutta(gauss2_table),
    'Неявный метод Гаусса 4 порядка': implicit_runge_kutta(gauss4_table),
    'Адамс-Башфорт 4': adams_method(adams_bashforth_coeff, implicit=False),
    'Адамс-Моултон 4': adams_method(adams_moulton_coeff, implicit=True),
}

# Вычисляем аналитическое решение
exact_sol = exact_solution(x)

max_errors = {}

# ===== ВЫВОД РЕЗУЛЬТАТОВ =====
print("ЗАДАЧА КОШИ")
print("=" * 70)
print(f"Уравнение: x³y' - y² = x⁴")
print(f"y({x0}) = {y0:.6f}, x ∈ [{x0}; {x_end}], h = {h}")
print()

print("СВОДНАЯ ТАБЛИЦА МАКСИМАЛЬНЫХ ПОГРЕШНОСТЕЙ")
print("=" * 70)
print(f"{'Метод':<40} | {'Порядок':^8} | {'Макс. погр.':>12}")
print("-" * 70)

orders = {
    'Явный метод Эйлера': 1,
    'Метод Хойна 2 порядка': 2,
    'Метод Хойна 3 порядка': 3,
    'Классический RK4': 4,
    'Мерсон 3(4)': 4,
    'Неявный метод Гаусса 2 порядка': 2,
    'Неявный метод Гаусса 4 порядка': 4,
    'Адамс-Башфорт 4': 4,
    'Адамс-Моултон 4': 4
}

for name, solution in methods.items():
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    max_error = max(error)
    max_errors[name] = max_error

    order = orders[name]
    method_name = name.ljust(40)
    order_str = str(order).center(8)
    error_str = f"{max_error:.10f}".rjust(18)

    print(f"{method_name} | {order_str} | {error_str}")

print("=" * 70)
print()

# ===== УЛУЧШЕННЫЕ ГРАФИКИ =====
fig = plt.figure(figsize=(16, 10))

# График 1: Сравнение только низких порядков (1 и 2) с точным решением
ax1 = plt.subplot(2, 2, 1)
plt.plot(x, exact_sol, 'k-', linewidth=2.5, label='Аналитическое решение', zorder=5)
plt.plot(x, methods['Явный метод Эйлера'], 'o-', color='red', markersize=4,
         label='Эйлер (порядок 1)', linewidth=1.5)
plt.plot(x, methods['Метод Хойна 2 порядка'], 's-', color='blue', markersize=4,
         label='Хойн 2 (порядок 2)', linewidth=1.5)
plt.plot(x, methods['Неявный метод Гаусса 2 порядка'], '^-', color='green', markersize=4,
         label='Гаусс 2 (порядок 2)', linewidth=1.5)
plt.xlabel('x', fontsize=11)
plt.ylabel('y(x)', fontsize=11)
plt.title('Методы низких порядков (1-2)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# График 2: Сравнение методов высоких порядков (3-4) - увеличенная область
ax2 = plt.subplot(2, 2, 2)
# Увеличиваем небольшую область для видимости различий
x_zoom_start = 3.0
x_zoom_end = 4.0
zoom_indices = [i for i, val in enumerate(x) if x_zoom_start <= val <= x_zoom_end]

if zoom_indices:
    x_zoom = [x[i] for i in zoom_indices]
    exact_zoom = [exact_sol[i] for i in zoom_indices]

    plt.plot(x_zoom, exact_zoom, 'k-', linewidth=2.5, label='Аналитическое', zorder=5)

    high_order_methods = ['Метод Хойна 3 порядка', 'Классический RK4',
                          'Мерсон 3(4)', 'Неявный метод Гаусса 4 порядка',
                          'Адамс-Башфорт 4', 'Адамс-Моултон 4']
    colors_high = ['green', 'orange', 'purple', 'brown', 'pink', 'olive']

    for name, color in zip(high_order_methods, colors_high):
        sol_zoom = [methods[name][i] for i in zoom_indices]
        plt.plot(x_zoom, sol_zoom, 'o-', color=color, markersize=5,
                 label=name.replace(' порядка', '').replace('Метод ', ''),
                 linewidth=1.5, alpha=0.8)

    plt.xlabel('x', fontsize=11)
    plt.ylabel('y(x)', fontsize=11)
    plt.title(f'Методы высоких порядков (3-4)\nУвеличенная область x ∈ [{x_zoom_start}, {x_zoom_end}]',
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

# График 3: Логарифм погрешности для всех методов
ax3 = plt.subplot(2, 2, 3)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
for (name, solution), color in zip(methods.items(), colors):
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    error = [max(e, 1e-16) for e in error]
    short_name = name.replace(' порядка', '').replace('Неявный метод ', 'Неявн. ')
    plt.semilogy(x, error, 'o-', color=color, label=short_name,
                 markersize=4, linewidth=1.5)

plt.xlabel('x', fontsize=11)
plt.ylabel('|Погрешность| (логарифм)', fontsize=11)
plt.title('Погрешность всех методов (логарифмическая шкала)', fontsize=12, fontweight='bold')
plt.legend(fontsize=9, ncol=2)
plt.grid(True, alpha=0.3, which='both')

# График 4: Сравнение погрешностей по порядкам
ax4 = plt.subplot(2, 2, 4)
order_groups = {
    1: ['Явный метод Эйлера'],
    2: ['Метод Хойна 2 порядка', 'Неявный метод Гаусса 2 порядка'],
    3: ['Метод Хойна 3 порядка'],
    4: ['Классический RK4', 'Мерсон 3(4)', 'Неявный метод Гаусса 4 порядка',
        'Адамс-Башфорт 4', 'Адамс-Моултон 4']
}

bar_positions = []
bar_values = []
bar_colors = []
bar_labels = []

pos = 0
color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}

for order in sorted(order_groups.keys()):
    for method in order_groups[order]:
        bar_positions.append(pos)
        bar_values.append(max_errors[method])
        bar_colors.append(color_map[order])
        short_label = method.replace(' порядка', '').replace('Неявный метод ', 'Неявн. ')
        bar_labels.append(f"{short_label}\n(p={order})")
        pos += 1
    pos += 0.5  # Gap between order groups

plt.bar(bar_positions, bar_values, color=bar_colors, alpha=0.7, edgecolor='black')
plt.xticks(bar_positions, bar_labels, rotation=45, ha='right', fontsize=8)
plt.yscale('log')
plt.ylabel('Максимальная погрешность (логарифм)', fontsize=11)
plt.title('Сравнение максимальных погрешностей по методам', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y', which='both')

plt.tight_layout()
plt.show()

# ===== ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ПОГРЕШНОСТЕЙ =====
print("\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ПОГРЕШНОСТЕЙ")
print("=" * 85)
print(f"{'Метод':<40} | {'Средняя погр.':>13} | {'Погр. в конце':>13}")
print("-" * 85)

for name, solution in methods.items():
    error = [abs(a - b) for a, b in zip(solution, exact_sol)]
    mean_error = np.mean(error)
    final_error = error[-1]

    method_name = name.ljust(40)
    mean_str = f"{mean_error:.10f}".center(18)
    final_str = f"{final_error:.10f}".center(18)

    print(f"{method_name} | {mean_str} | {final_str}")

print("=" * 85)

print("\nОБЪЯСНЕНИЕ:")
print("-" * 85)
print("Методы высоких порядков (3-4) дают очень близкие результаты при данном шаге h=0.20.")
print("Их погрешности отличаются на несколько порядков от методов низких порядков (1-2).")
print("Для лучшей визуализации различий между методами высоких порядков:")
print("  • График 2 показывает увеличенную область решения")
print("  • График 3 использует логарифмическую шкалу для погрешностей")
print("  • График 4 сравнивает максимальные погрешности в виде столбчатой диаграммы")