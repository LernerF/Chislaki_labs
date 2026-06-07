import numpy as np
import matplotlib.pyplot as plt

# Данные из таблицы
xs = np.array([-3.32, -2.64, -1.96, -1.28, -0.60, 0.08, 0.76, 1.44, 2.12, 2.80])
ys = np.array([-1.6753, -0.0521, 0.2858, 1.0675, 1.3427, 1.9528, 1.1032, 1.1243, 0.4318, -1.4219])
x_star = -0.824


def fit_least_squares(xs, ys, degree):
    """Метод наименьших квадратов для аппроксимации многочленом заданной степени"""
    n = len(xs)
    m = degree + 1

    # Вычисление сумм степеней x
    S = np.zeros(2 * degree + 1)
    for k in range(2 * degree + 1):
        S[k] = np.sum(xs ** k)

    # Построение матрицы системы
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            A[i, j] = S[i + j]

    # Построение правой части
    b = np.zeros(m)
    for i in range(m):
        b[i] = np.sum(ys * (xs ** i))

    # Решение системы
    a = solve_linear_system(A, b)
    return a


def solve_linear_system(A_orig, b_orig):
    """Решение линейной системы методом Гаусса с выбором главного элемента"""
    n = len(b_orig)
    A = A_orig.copy()
    b = b_orig.copy()

    for k in range(n):
        # Выбор главного элемента
        pivot = k
        max_val = abs(A[k, k])
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                pivot = i

        if pivot != k:
            # Перестановка строк
            A[[k, pivot]] = A[[pivot, k]]
            b[k], b[pivot] = b[pivot], b[k]

        if abs(A[k, k]) < 1e-15:
            continue

        # Исключение
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i, j] * x[j]
        x[i] = s / A[i, i] if abs(A[i, i]) > 1e-15 else 0.0

    return x


def evaluate_polynomial(coeffs, x):
    """Вычисление значения многочлена в точке x"""
    result = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def sum_squared_errors(xs, ys, coeffs):
    """Вычисление суммы квадратов ошибок"""
    sse = 0.0
    for i in range(len(xs)):
        y_pred = evaluate_polynomial(coeffs, xs[i])
        error = ys[i] - y_pred
        sse += error ** 2
    return sse


def print_coefficients(coeffs, degree):
    """Вывод коэффициентов многочлена"""
    print(f"Коэффициенты многочлена степени {degree}:")
    for i, coeff in enumerate(coeffs):
        print(f"  a{i} = {coeff:.10f}")
    print()


# ========== ОСНОВНЫЕ ВЫЧИСЛЕНИЯ ==========
print("=" * 80)
print("МЕТОД НАИМЕНЬШИХ КВАДРАТОВ")
print("АППРОКСИМАЦИЯ МНОГОЧЛЕНАМИ 1-Й, 2-Й И 3-Й СТЕПЕНИ")
print("=" * 80)

results = []

for degree in range(1, 4):
    coeffs = fit_least_squares(xs, ys, degree)
    sse = sum_squared_errors(xs, ys, coeffs)
    rmse = np.sqrt(sse / len(xs))
    value_at_star = evaluate_polynomial(coeffs, x_star)

    results.append({
        'degree': degree,
        'coeffs': coeffs,
        'sse': sse,
        'rmse': rmse,
        'value': value_at_star
    })

    print(f"\nМногочлен степени {degree}:")
    print_coefficients(coeffs, degree)
    print(f"Сумма квадратов ошибок: {sse:.6f}")
    print(f"Среднеквадратичное отклонение: {rmse:.6f}")
    print(f"Значение в точке x* = {x_star}: {value_at_star:.6f}")

# Создание фигуры
fig = plt.figure(figsize=(16, 10), dpi=100)
try:
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
except:
    pass

# Подготовка данных для графиков
x_plot = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 400)
y_plots = []

for result in results:
    y_plot = np.array([evaluate_polynomial(result['coeffs'], x) for x in x_plot])
    y_plots.append(y_plot)

# Основной график: все многочлены и исходные данные
plt.subplot(2, 2, 1)
plt.plot(xs, ys, 'bo-', linewidth=2, markersize=8, label='Исходные данные', zorder=5)
plt.plot(x_plot, y_plots[0], 'r-', linewidth=2, label=f'1-я степень (RMSE={results[0]["rmse"]:.4f})')
plt.plot(x_plot, y_plots[1], 'g-', linewidth=2, label=f'2-я степень (RMSE={results[1]["rmse"]:.4f})')
plt.plot(x_plot, y_plots[2], 'm-', linewidth=2, label=f'3-я степень (RMSE={results[2]["rmse"]:.4f})')

# Точка x*
plt.axvline(x=x_star, color='purple', linestyle='--', alpha=0.7, label=f'x* = {x_star}')
for i, result in enumerate(results):
    color = ['red', 'green', 'magenta'][i]
    plt.plot(x_star, result['value'], 'o', color=color, markersize=8,
             label=f'P{result["degree"]}(x*) = {result["value"]:.4f}', zorder=6)

plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('y', fontsize=12, fontweight='bold')
plt.title('Аппроксимация методом наименьших квадратов', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# График 2: Ошибки аппроксимации
plt.subplot(2, 2, 2)
for i, result in enumerate(results):
    y_pred = np.array([evaluate_polynomial(result['coeffs'], x) for x in xs])
    errors = ys - y_pred
    color = ['red', 'green', 'magenta'][i]
    plt.plot(xs, errors, 'o-', color=color, linewidth=2, markersize=6,
             label=f'Ошибки {result["degree"]}-й степени')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('Ошибка', fontsize=12, fontweight='bold')
plt.title('Ошибки аппроксимации', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# График 3: Сравнение среднеквадратических ошибок
plt.subplot(2, 2, 3)
degrees = [result['degree'] for result in results]
rmses = [result['rmse'] for result in results]
colors = ['red', 'green', 'magenta']

bars = plt.bar(degrees, rmses, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Степень многочлена', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Среднеквадратичная ошибка по степеням', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Добавление значений на столбцы
for bar, rmse in zip(bars, rmses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')

# График 4: Значения в точке x*
plt.subplot(2, 2, 4)
values_at_star = [result['value'] for result in results]

plt.plot(degrees, values_at_star, 'bo-', linewidth=2, markersize=8, label='P(x*)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Находим значение исходной функции в ближайшей точке к x*
idx_near = np.argmin(np.abs(xs - x_star))
y_near = ys[idx_near]
plt.axhline(y=y_near, color='red', linestyle='--', alpha=0.7,
            label=f'Ближайшее значение данных: {y_near:.4f}')

plt.xlabel('Степень многочлена', fontsize=12, fontweight='bold')
plt.ylabel('P(x*)', fontsize=12, fontweight='bold')
plt.title('Значения аппроксимирующих функций в точке x*', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ========== ИТОГОВАЯ ТАБЛИЦА ==========
print("=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)
print(f"{'Степень':<8} {'P(x*)':<12} {'SSE':<15} {'RMSE':<15}")
print("-" * 50)
for result in results:
    print(f"{result['degree']:<8} {result['value']:<12.6f} {result['sse']:<15.6f} {result['rmse']:<15.6f}")