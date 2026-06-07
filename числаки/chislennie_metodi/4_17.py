import math
import matplotlib.pyplot as plt


def analytic_solution(x):
    """Аналитическое решение: y(x) = (x/e)^x + (e/x)^x"""
    e = math.e
    return (x/e)**x + (e/x)**x

def build_grid(a, b_val, h):
    """Построение сетки узлов"""
    N = int((b_val - a) / h) + 1
    x = [a + i * h for i in range(N)]
    return x, N

def zero_matrix(n):
    """Создание нулевой матрицы"""
    return [[0.0 for _ in range(n)] for _ in range(n)]

def zero_vector(n):
    """Создание нулевого вектора"""
    return [0.0 for _ in range(n)]

# ========== НАЧАЛЬНЫЕ ПРЕОБРАЗОВАНИЯ ==========
# Исходное уравнение: xy'' ln(x) - y' - xy ln³(x) = 0
#
# Приводим к стандартному виду y'' + p(x)y' + q(x)y = f(x)
# Делим все на x·ln(x):
# y'' - (1/(x·ln(x)))y' - ln²(x)·y = 0
#
# Получаем:
# p(x) = -1/(x·ln(x))
# q(x) = -ln²(x)
# f(x) = 0
# ==============================================

def p_func(x):
    """Коэффициент p(x) = -1/(x·ln(x))"""
    return -1.0 / (x * math.log(x))

def q_func(x):
    """Коэффициент q(x) = -ln²(x)"""
    ln_x = math.log(x)
    return -ln_x * ln_x

def f_func(x):
    """Правая часть f(x) = 0"""
    return 0.0

def build_matrix_and_rhs_first_order(x, h):
    """Построение системы с аппроксимацией 1-го порядка"""
    N = len(x)
    A = zero_matrix(N)
    b = zero_vector(N)

    # Левая граница: y(1.2) - y'(1.2) = 3.645
    # Аппроксимация 1-го порядка: y'(x₀) ≈ (y₁ - y₀)/h
    # y₀ - (y₁ - y₀)/h = 3.645
    # (1 + 1/h)y₀ - (1/h)y₁ = 3.645
    A[0][0] = 1.0 + 1.0/h
    A[0][1] = -1.0/h
    b[0] = 3.645

    # Правая граница: y(4.5) = 9.899
    A[N-1][N-1] = 1.0
    b[N-1] = 9.899

    # Внутренние узлы
    # y_{k-1}(1 - p(xₖ)h/2) + yₖ(-2 + h²q(xₖ)) + y_{k+1}(1 + p(xₖ)h/2) = h²f(xₖ)
    for k in range(1, N-1):
        xx = x[k]
        p = p_func(xx)
        q = q_func(xx)
        f = f_func(xx)

        A[k][k-1] = 1.0 - (h * p) / 2.0
        A[k][k] = -2.0 + h*h * q
        A[k][k+1] = 1.0 + (h * p) / 2.0
        b[k] = h*h * f

    return A, b

def build_matrix_and_rhs_second_order(x, h):
    """Построение системы с аппроксимацией 2-го порядка"""
    N = len(x)
    A = zero_matrix(N)
    b = zero_vector(N)

    # Левая граница: y(1.2) - y'(1.2) = 3.645
    # Аппроксимация 2-го порядка: y'(x₀) ≈ (-3y₀ + 4y₁ - y₂)/(2h)
    # y₀ - (-3y₀ + 4y₁ - y₂)/(2h) = 3.645
    # (2h + 3)y₀ - 4y₁ + y₂ = 7.29h
    A[0][0] = 2.0*h + 3.0
    A[0][1] = -4.0
    A[0][2] = 1.0
    b[0] = 7.29 * h

    # Правая граница: y(4.5) = 9.899
    A[N-1][N-1] = 1.0
    b[N-1] = 9.899

    # Внутренние узлы (те же, что и для 1-го порядка)
    for k in range(1, N-1):
        xx = x[k]
        p = p_func(xx)
        q = q_func(xx)
        f = f_func(xx)

        A[k][k-1] = 1.0 - (h * p) / 2.0
        A[k][k] = -2.0 + h*h * q
        A[k][k+1] = 1.0 + (h * p) / 2.0
        b[k] = h*h * f

    return A, b

def lu_decomposition(A):
    """LU-разложение матрицы"""
    n = len(A)
    L = zero_matrix(n)
    U = zero_matrix(n)

    for i in range(n):
        # Вычисление U
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - s

        # Вычисление L
        L[i][i] = 1.0
        for j in range(i+1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U

def lu_solve(L, U, b):
    """Решение системы Ly = b, Ux = y"""
    n = len(b)

    # Прямая подстановка: Ly = b
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i][k] * y[k]
        y[i] = b[i] - s

    # Обратная подстановка: Ux = y
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        s = 0.0
        for k in range(i+1, n):
            s += U[i][k] * x[k]
        x[i] = (y[i] - s) / U[i][i]

    return x

# ========== ОСНОВНАЯ ПРОГРАММА ==========

# Параметры задачи
a, b_val = 1.2, 4.5
h = 0.3

# Построение сетки
x, N = build_grid(a, b_val, h)

print(f"Краевая задача: xy'' ln(x) - y' - xy ln³(x) = 0")
print(f"Граничные условия: y(1.2) - y'(1.2) = 3.645, y(4.5) = 9.899")
print(f"Отрезок: [{a}, {b_val}], шаг h = {h}")
print(f"Количество узлов: N = {N}")
print(f"\nСетка узлов: {[round(xi, 2) for xi in x]}")
print()

# Решение с аппроксимацией 1-го порядка
print("=" * 60)
print("РЕШЕНИЕ С АППРОКСИМАЦИЕЙ 1-ГО ПОРЯДКА")
print("=" * 60)
A1, b1 = build_matrix_and_rhs_first_order(x, h)
L1, U1 = lu_decomposition(A1)
y1 = lu_solve(L1, U1, b1)

# Решение с аппроксимацией 2-го порядка
print("\nРЕШЕНИЕ С АППРОКСИМАЦИЕЙ 2-ГО ПОРЯДКА")
print("=" * 60)
A2, b2 = build_matrix_and_rhs_second_order(x, h)
L2, U2 = lu_decomposition(A2)
y2 = lu_solve(L2, U2, b2)

# Аналитическое решение
y_analytic = [analytic_solution(xk) for xk in x]

# Таблица решений
print("\n" + "=" * 80)
print("ТАБЛИЦА ЧИСЛЕННЫХ РЕШЕНИЙ")
print("=" * 80)
print(f"{'k':>3} | {'x':>6} | {'y (1-й пор)':>12} | {'y (2-й пор)':>12}")
print("-" * 80)
for k in range(N):
    print(f"{k:3d} | {x[k]:6.2f} | {y1[k]:12.6f} | {y2[k]:12.6f}")

# Сравнение с аналитикой
print("\n" + "=" * 100)
print("СРАВНЕНИЕ С АНАЛИТИЧЕСКИМ РЕШЕНИЕМ")
print("=" * 100)
print(f"{'k':>3} | {'x':>6} | {'y_аналит':>12} | {'y_1пор':>12} | {'погр_1':>10} | {'y_2пор':>12} | {'погр_2':>10}")
print("-" * 100)

max_err1 = 0.0
max_err2 = 0.0
for k in range(N):
    err1 = abs(y_analytic[k] - y1[k])
    err2 = abs(y_analytic[k] - y2[k])
    if err1 > max_err1:
        max_err1 = err1
    if err2 > max_err2:
        max_err2 = err2
    print(f"{k:3d} | {x[k]:6.2f} | {y_analytic[k]:12.6f} | {y1[k]:12.6f} | {err1:10.6f} | {y2[k]:12.6f} | {err2:10.6f}")

print("\n" + "=" * 100)
print(f"Максимальная погрешность (1-й порядок): {max_err1:.8f}")
print(f"Максимальная погрешность (2-й порядок): {max_err2:.8f}")
print("=" * 100)

# Построение графиков
plt.figure(figsize=(12, 6))

# График 1: Все решения
plt.subplot(1, 2, 1)
plt.plot(x, y_analytic, 'k-', linewidth=2, label='Аналитическое', marker='o')
plt.plot(x, y1, 'r--', linewidth=1.5, label='1-й порядок', marker='s')
plt.plot(x, y2, 'b-.', linewidth=1.5, label='2-й порядок', marker='^')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.title('Решение краевой задачи: xy\'\' ln(x) - y\' - xy ln³(x) = 0', fontsize=11)
plt.grid(True, alpha=0.3)

# График 2: Погрешности
plt.subplot(1, 2, 2)
errors1 = [abs(y_analytic[k] - y1[k]) for k in range(N)]
errors2 = [abs(y_analytic[k] - y2[k]) for k in range(N)]
plt.plot(x, errors1, 'r-o', linewidth=1.5, label='Погрешность 1-го порядка')
plt.plot(x, errors2, 'b-^', linewidth=1.5, label='Погрешность 2-го порядка')
plt.xlabel('x', fontsize=12)
plt.ylabel('Абсолютная погрешность', fontsize=12)
plt.legend(fontsize=10)
plt.title('Погрешности численных методов', fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()