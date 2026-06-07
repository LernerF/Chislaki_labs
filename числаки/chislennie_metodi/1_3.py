# расширенная матрица [A|b]

matrix = [
    [6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.0],
    [-7.0, 9.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
    [0.0, 2.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0, -25.0],
    [0.0, 0.0, -4.0, 12.0, -7.0, 0.0, 0.0, 0.0, 30.0],
    [0.0, 0.0, 0.0, 6.0, 9.0, 3.0, 0.0, 0.0, 33.0],
    [0.0, 0.0, 0.0, 0.0, -5.0, 11.0, 4.0, 0.0, 7.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, -9.0, 3.0, -55.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 7.0, 56.0]
]

n = len(matrix)

def get_A_b(aug):
    A = [[float(aug[i][j]) for j in range(n)] for i in range(n)]
    b = [float(aug[i][n]) for i in range(n)]
    return A, b

def from_full_matrix_to_tridiagonal(A, b):
    main = [A[i][i] for i in range(n)]
    lower = [A[i][i-1] for i in range(1, n)]
    upper = [A[i][i+1] for i in range(0, n-1)]
    rhs = [float(v) for v in b]
    return lower, main, upper, rhs

def stability_check(lower, main, upper):
    ok = True
    print("Проверка устойчивости (диагонального преобладания):\n")
    for i in range(len(main)):
        bi = abs(main[i])
        ai = abs(lower[i-1]) if i > 0 else 0.0
        ci = abs(upper[i]) if i < len(main)-1 else 0.0
        lhs = bi
        rhs = ai + ci
        result = lhs >= rhs
        print(f"Строка {i+1}: |b{i+1}| = {lhs:.6f} >= |a{i+1}| + |c{i+1}| = {rhs:.6f} : {result}")
        if not result:
            ok = False

    if main[0] != 0:
        c1_b1 = abs(upper[0]) / abs(main[0])
    else:
        c1_b1 = float('inf')
    if main[-1] != 0:
        an_bn = abs(lower[-1]) / abs(main[-1])
    else:
        an_bn = float('inf')

    print(f"\nПроверка частных условий: c1/b1 < 1 -> {c1_b1:.6f} < 1 : {c1_b1 < 1}")
    print(f"                         a_n/b_n < 1 -> {an_bn:.6f} < 1 : {an_bn < 1}")

    if not (c1_b1 < 1 and an_bn < 1):
        ok = False

    return ok

# Прямой ход
def compute_P_Q(lower, main, upper, rhs):
    n = len(main)
    a = [0.0]*n
    b = [float(v) for v in main]
    c = [0.0]*n
    d = [float(v) for v in rhs]
    for i in range(1, n):
        a[i] = float(lower[i-1])
    for i in range(n-1):
        c[i] = float(upper[i])
    P = [0.0]*n
    Q = [0.0]*n
    denoms = [0.0]*n
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    denoms[0] = b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i-1]
        P[i] = (-c[i] / denom) if i < n-1 else 0.0
        Q[i] = (d[i] - a[i] * Q[i-1]) / denom
        denoms[i] = denom
    return P, Q, denoms

# Обратный ход
def back_substitution(P, Q):
    n = len(Q)
    x = [0.0]*n
    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i] * x[i+1] + Q[i]
    return x

# Определитель
def determinant_from_denoms(denoms):
    det = 1.0
    for v in denoms:
        det *= v
    return det

def fmt_value_two(v):
    return f"{v:.2f}"


A, b = get_A_b(matrix)
lower, main, upper, rhs = from_full_matrix_to_tridiagonal(A, b)


stable = stability_check(lower, main, upper)
print()
if stable:
    print("Условие устойчивости выполнено: метод прогонки устойчив.\n")
else:
    print("Внимание: условие устойчивости НЕ выполнено — метод прогонки может быть неустойчив.\n")

P, Q, denoms = compute_P_Q(lower, main, upper, rhs)
x = back_substitution(P, Q)
detA = determinant_from_denoms(denoms)

print("Решение системы:")
for i, val in enumerate(x, 1):
    print(f"x{i} = {fmt_value_two(val)}")

print()
print("Определитель матрицы:")
print(f"det(A) = {fmt_value_two(detA)}")
