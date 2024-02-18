import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de entrada
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

reg = LinearRegression().fit(X, y)

# Realizar previsão para novos valores
novos_valores = np.array([[6], [7]])
previsao = reg.predict(novos_valores)

print('O resultado da previsão é: \n')
print(previsao)

# Vamos ver o exemplo através de um gráfico para melhor entendimento

#Importando a biblioteca necessária
import matplotlib.pyplot as plt

print("O resultado gráfico desta Regressão Linear é:")
print() # Este print sem argumento gera uma linha em branco

# Plotar o gráfico de dispersão e a linha de regressão
plt.scatter(X, y, color='black')
plt.plot(X, reg.predict(X), color='blue', linewidth=3, label='Regressão Linear')

# Plotar a previsão como uma nova linha
plt.plot(novos_valores, previsao, color='green', linestyle='--', linewidth=3, label='Previsão')

# Configurar as legendas do gráfico
plt.title('Regressão linear simples')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.show()