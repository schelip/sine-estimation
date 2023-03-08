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
    = 5i + 1 multiplicações

    Args:
        x (float): valor em rad do ângulo para estimar o seno
        n (int): ordem da Série de Maclaurin

    Returns:
        float: uma estimativa de sin x
    """
    result = 0
    for i in range(math.ceil(n / 2)):
        result += (pow(-1, i) / math.factorial(2 * i + 1)) * pow(x, 2 * i + 1)
    return result

def sin_taylor_opt(x, n):
    """Versão otimizada da função para estimar sin x usando
    uma Série de Maclaurin com multiplicações reduzidas.

    Os valores da última iteração são salvos em memória e reutilizados
    na próxima iteração. Assim, toda iteração i realiza exatamente 5 multiplicações.

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

if __name__ == '__main__':
    x_vals = np.linspace(0, 2 * math.pi, 1000)
    y_taylor = np.array([sin_taylor_opt(x, 11) for x in x_vals])
    y_exact = np.array([math.sin(x) for x in x_vals])

    plt.plot(x_vals, y_taylor, label='Taylor')
    plt.plot(x_vals, y_exact, label='Exact')
    plt.legend()
    plt.show()