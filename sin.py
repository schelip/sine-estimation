import math, timeit
import numpy as np
import matplotlib.pyplot as plt

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

def sin_taylor_opt(x, n):
    """Versão otimizada da função para estimar sin x usando
    uma Série de Maclaurin com multiplicações reduzidas.

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

def sin_pade_unopt(x: float, n: int, m: int, p: int):
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
    a_coef = coef[4:]
    b_coef = coef[:4]
    a = 0
    b = 1
    for i in range(n + 1):
        a += a_coef[i] * (x ** i)
    for i in range(m):
        b += b_coef[i] * (x ** i)
    return a / b

if __name__ == '__main__':
    x_vals = np.linspace(0, 2 * math.pi, 1000)
    y_taylor = np.array([sin_taylor_opt(x, 11) for x in x_vals])
    y_exact = np.array([math.sin(x) for x in x_vals])
    y_pade = np.array([sin_pade_unopt(x, 7, 4, 11) for x in x_vals])

    plt.plot(x_vals, y_taylor, label='Taylor')
    plt.plot(x_vals, y_exact, label='Exact')
    plt.plot(x_vals, y_pade, label='Pade')
    plt.legend()
    plt.show()