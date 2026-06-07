matrix = [
    [2, -5, -4, 7, 4, 13],
    [5, 2, 1, -5, 2, 43],
    [4, 9, 7, -4, -5, 24],
    [2, 5, -4, 9, 3, 30],
    [8, 3, -5, -6, -2, 81]
]

n = len(matrix)  # размерность системы без b 


def solve_gauss(mat):
    """Решение СЛАУ методом Гаусса"""
    A = [row[:] for row in mat]  # создаём копию матрицы

    # ПРЯМОЙ ХОД: приведение к треугольному виду
    for i in range(n):            # i - номер шага (диагональный элемент)
        for k in range(i + 1, n):  # k - строки ниже текущей
            factor = A[k][i] / A[i][i]  # множитель для обнуления
            for j in range(i, n + 1):   # j - столбцы (включая правую часть)
                A[k][j] -= factor * A[i][j]  # вычитаем строку i из строки k

    # ОБРАТНЫЙ ХОД: вычисление неизвестных (снизу вверх)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):        # идём с последней строки к первой
        x[i] = A[i][n]                     # берём правую часть
        for j in range(i + 1, n):          # вычитаем уже найденные x
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]                     # делим на диагональный элемент

    return x


def get_determinant(mat):
    """Вычисление определителя методом Гаусса"""
    A = [row[:] for row in mat]
    det = 1.0

    # Прямой ход (без правой части)
    for i in range(n):
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):          # идём только до n (без правой части)
                A[k][j] -= factor * A[i][j]
        det *= A[i][i]                       # опр - произведение диагональных элементов

    return det


def get_inverse(mat):
    """Вычисление обратной матрицы методом Гаусса-Жордана с выбором главного элемента"""
    # Преобразуем в float для точности
    A = [[float(mat[i][j]) for j in range(n)] for i in range(n)]
    row_order = list(range(n))  # для отслеживания перестановок строк

    # СОЗДАЁМ РАСШИРЕННУЮ МАТРИЦУ [A | I]
    augmented = []
    for i in range(n):
        row = A[i][:] + [1.0 if j == i else 0.0 for j in range(n)]
        augmented.append(row)  # теперь матрица размера n × 2n

    # ПРЯМОЙ ХОД С ВЫБОРОМ ГЛАВНОГО ЭЛЕМЕНТА
    for i in range(n):
        # Ищем максимальный элемент в столбце i (для устойчивости)
        max_row = i
        for j in range(i + 1, n):
            if abs(augmented[row_order[j]][i]) > abs(augmented[row_order[max_row]][i]):
                max_row = j

        # Если нашли строку с бОльшим элементом - меняем порядок
        if max_row != i:
            row_order[i], row_order[max_row] = row_order[max_row], row_order[i]

        pivot = augmented[row_order[i]][i]  # опорный элемент
        # Обнуляем элементы НИЖЕ диагонали
        for j in range(i + 1, n):
            factor = augmented[row_order[j]][i] / pivot
            for k in range(i, 2 * n):
                augmented[row_order[j]][k] -= factor * augmented[row_order[i]][k]

    # ОБРАТНЫЙ ХОД (превращаем левую часть в единичную матрицу)
    for i in range(n - 1, -1, -1):
        pivot = augmented[row_order[i]][i]
        # Нормализуем текущую строку (делим на диагональный элемент)
        for k in range(i, 2 * n):
            augmented[row_order[i]][k] /= pivot
        # Обнуляем элементы ВЫШЕ диагонали
        for j in range(i - 1, -1, -1):
            factor = augmented[row_order[j]][i]
            for k in range(i, 2 * n):
                augmented[row_order[j]][k] -= factor * augmented[row_order[i]][k]

    # ИЗВЛЕКАЕМ ОБРАТНУЮ МАТРИЦУ (последние n столбцов)
    inv_A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inv_A[i][j] = augmented[row_order[i]][n + j]

    return inv_A


def multiply_matrices(A, B):
    """Умножение двух квадратных матриц"""
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]  # строка на столбец
            C[i][j] = s
    return C


def multiply_matrix_vector(A, x):
    """Умножение матрицы на вектор"""
    res = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]  # сумма по строке
        res[i] = s
    return res



# 1. Решение системы методом Гаусса
print("Решение системы:")
solution = solve_gauss(matrix)
for i, x in enumerate(solution, 1):
    print(f"x{i} = {x:.2f}")

# 2. Вычисление определителя
print(f"\nОпределитель матрицы:")
det = get_determinant(matrix)
print(f"det(A) = {det:.2f}")

# 3. Вычисление обратной матрицы
print(f"\nОбратная матрица A^(-1):")
inv = get_inverse(matrix)
for row in inv:
    print("[" + ", ".join(f"{x:10.6f}" for x in row) + "]")

# 4. ПРОВЕРКА: A * A^(-1) должна равняться единичной матрице E
A_only = [row[:n] for row in matrix]  # берём только матрицу A (без правой части)
prod = multiply_matrices(A_only, inv)

print("\nПроверка A * A^(-1) = E:")
for row in prod:
    print("[" + ", ".join(f"{x: .6f}" for x in row) + "]")

# 5. ПРОВЕРКА: A * x должна равняться вектору b
Ax = multiply_matrix_vector(A_only, solution)
b_vec = [row[n] for row in matrix]

print("\nПроверка A * x = b:")
for i in range(n):
    print(f"  (A*x)_{i+1} = {Ax[i]:.6f}")