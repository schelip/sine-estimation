import math, timeit
import numpy as np
import matplotlib.pyplot as plt

global taylor_11, pade_7_4_11
taylor_11 = np.array([
        1.00000000e+00, -1.66666667e-01,
        8.33333333e-03,  -1.98412698e-04,
        2.75573192e-06,  -2.50521084e-08])
pade_7_4_11 = np.array([
        2.06060606e-02, 1.59932660e-04,
        1.00000000e+00, -1.46060606e-01,
        5.05892256e-03, -5.33509700e-05])

def sin_taylor_unopt(x: float, n: int) -> float:
    """Versão simples da função para estimar sin x usando
    uma Série de Maclaurin.

    A iteração i realiza:
        i multiplicações para positivar/negativar o termo;
        2i multiplicações para calcular o fatorial;
        2i multiplicações para calcular a potência;
        1 multiplicação entre os fatores do termo;
        1 divisão entre os fatores do termo;
    = 5i + 2 operações

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    for i in range(math.ceil(n / 2)):
        result += (((-1) ** i) / math.factorial(2 * i + 1)) * (x ** (2 * i + 1))
    return result

def sin_taylor_opt(x: float, n: int) -> float:
    """Versão otimizada com multiplicações reduzidas
    da função para estimar sin x usando uma Série de Maclaurin.

    Os valores da última iteração são salvos em memória e reutilizados
    na próxima iteração. Assim, toda iteração i realiza exatamente
    6 operações de multiplicação/divisão.

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    factorial = 1 
    factor = 1
    positive = 1
    power = x
    for _ in range(math.ceil(n / 2)):
        result += (positive / factorial) * power
        positive = -positive
        factor += 1
        factorial *= factor
        factor += 1
        factorial *= factor
        power *= x * x
    return result

def sin_taylor_predef(x: float) -> float:
    """Versão utilizando coeficientes predefinidos
    da função para estimar sin x usando uma Série de Maclaurin de ordem 11 (6 termos).

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    global taylor_11
    power = x
    result = 0
    for f in taylor_11:
        result += power * f
        power *= x * x
    return result

def sin_pade_unopt(x: float, n: int, m: int, p: int) -> float:
    """Versão simples da função para estimar sin x usando
    a aproximação de Padè.

    Calcula os coeficientes pra equação utilizando um sistema linear,
    igualando a uma série de Maclaurin.

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    p_coef = np.zeros(p + 1)
    for i in range(math.ceil(p / 2)):
        p_coef[2 * i + 1] = ((-1) ** i) / math.factorial(2 * i + 1)
    A = np.zeros((p + 1, n + m + 1))
    for i in range(p + 1):
        for j in range(min(i, m)):
            A[i, j] = - p_coef[i - j - 1]
        if m + i < p + 1:
            A[i, m + i] = 1
    coef = np.linalg.solve(A, p_coef)
    a_coef = coef[m:]
    b_coef = coef[:m]
    a = 0
    b = 1
    for i in range(n + 1):
        a += a_coef[i] * (x ** i)
    for i in range(m):
        b += b_coef[i] * (x ** i)
    return a / b

def sin_pade_predef(x: float) -> float:
    """Versão utilizando coeficientes predefinidos
    da função para estimar sin x usando uma estimativa de Padè de ordem 7, 4
    igualando com uma Série de Maclaurin de ordem 11 (6 termos).

    Args:
        x (float): valor em rad do ângulo para estimar o seno

    Returns:
        float: uma estimativa de sin x
    """
    global pade_7_4_11
    a_coef = pade_7_4_11[2:]
    b_coef = pade_7_4_11[:2]
    a = 0
    b = 1
    power = x
    for i in range(4):
        if i < 2:
            b += b_coef[i] * power
        a += a_coef[i] * power
        power *= x * x
    return a / b

if __name__ == '__main__':
    x_vals = np.linspace(0, 2 * math.pi, 1000)
    y_taylor = np.array([sin_taylor_predef(x) for x in x_vals])
    y_exact = np.array([math.sin(x) for x in x_vals])
    y_pade = np.array([sin_pade_predef(x) for x in x_vals])

    plt.plot(x_vals, y_taylor, label='Taylor')
    plt.plot(x_vals, y_exact, label='Exact')
    plt.plot(x_vals, y_pade, label='Pade')
    plt.legend()
    plt.show()