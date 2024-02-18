#@title 10.3.2) Exemplo prático com Scikit_learn - Modelo de Regressão linear (2)

# Importando a função make_regression do módulo sklearn.datasets.
# Essa função é utilizada para criar dados sintéticos para problemas de regressão.
from sklearn.datasets import make_regression

# Gerando uma massa de dados:
x, y = make_regression(n_samples=200, n_features=1, noise=30)
# Estamos gerando 200 amostras, cada uma com uma única feature (1 característica) e adicionando ruído de valor 30.
# O conjunto de dados gerado é armazenado nas variáveis x e y, onde x contém as features e y contém os valores alvo (rótulos).

# Importando a biblioteca de gráficos
import matplotlib.pyplot as plt

# Mostrando no gráfico:
# Criando um gráfico de dispersão (scatter). O gráfico mostra os pontos de dados do conjunto x em relação aos valores alvo y.
plt.scatter(x,y)
print("Modelo de Regressão Linear\n")
print("ETAPA 01: Gráfico de Dispersão com os dados gerados aleatóriamente para o nosso modelo de Regressão\n")
plt.show()  # Exibe o gráfico

# Aqui, estamos importando a classe LinearRegression do módulo sklearn.linear_model.
# Essa classe é usada para criar um modelo de regressão linear.
from sklearn.linear_model import LinearRegression
# Criação do modelo
modelo = LinearRegression()

# Aqui, estamos ajustando o modelo aos dados x e y.
# Ou seja, estamos treinando o modelo com os dados de entrada x e os valores alvo y.
modelo.fit(x,y)

# Nesta linha, estamos utilizando o modelo treinado para fazer previsões com base nos dados de entrada x.
# O resultado é uma previsão para cada ponto em x.
modelo.predict(x)

# Nesta linha, estamos novamente criando um gráfico de dispersão com os pontos de dados x em relação aos valores alvo y.
plt.scatter(x,y)
# Aqui, estamos usando a função plot do matplotlib.pyplot para traçar uma linha que representa as previsões feitas pelo modelo em relação aos dados x.
# A linha é desenhada em vermelho (color='red') com uma espessura de linha de 3 pixels (linewidth=3).
plt.plot(x, modelo.predict(x), color='red', linewidth=3)
print("ETAPA 02: Gráfico de Dispersão com a reta gerada pelo nosso modelo de regressão baseado nos pontos\n")
plt.show()  # Exibe o gráfico

