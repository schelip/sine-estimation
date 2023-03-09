# sin x estimation

Programa em python para comparar diferentes métodos de estimativa do seno de um valor x, além de aspectos como erros de aproximação por truncamento e tempo de execução.

Desenvolvido para o trabalho 2 da disciplina de Matemática Computacional com o Prof. Airton Marco no terceiro ano de Ciência da Computação na UEM.

## Uso
Para executar o programa, com `python3` instalado:
```bash
$ py .\sin.py <option>
```

Onde option é um de:
```
-t | --time-profile: imprime os resultados, tempo de execução e erro das funções de estimativa implementadas
-v | --plot-values: plota os resultados das funções de estimativa implementadas
-e | --plot-error: plota o erro das funções de estimativa implementadas
-x | --export: exporta os resultados das funções de estimativa implementadas e erro para um arquivo de texto
-h | --help: mostra esta informação de uso
```

O programa irá requisitar quais os limites para os valores com quais irá realizar os cálculos. Note que é possível utilizar `math.pi` ou simplesmente `pi` para representar o valor de π, além de outras operações matemáticas seguindo a sintaxe do python. Por exemplo:

```bash
$ py .\sin.py -e
Initial value [-pi/4]: - pi * 3
```
