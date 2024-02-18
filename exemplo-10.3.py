#@title 10.3.3) Exemplo prático com Scikit_learn - Clusterização (1)

# Este código é um exemplo simples de como usar a biblioteca Scikit-learn em Python para executar o algoritmo de clusterização K-means em um conjunto de dados.

# Primeiramente, o código importa as bibliotecas necessárias:
# NumPy para trabalhar com matrizes e o KMeans do Scikit-learn para executar o algoritmo de clusterização.

# Importando bibliotecas necessárias
import numpy as np
from sklearn.cluster import KMeans

# Depois, um conjunto de dados é definido como a variável X, que é uma matriz NumPy de seis pontos em um espaço bidimensional.

# Dados de entrada
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# O próximo passo é treinar o modelo de clusterização usando a função KMeans do Scikit-learn, que recebe como argumento o número de clusters que deseja-se identificar.
# Neste caso, o número de clusters é definido como 2.

# Treinar o modelo de clusterização
kmeans = KMeans(n_clusters=2).fit(X)

# Obter os rótulos de cluster para cada objeto
rotulos = kmeans.labels_

# Imprimir os rótulos de cluster
print("O resultado numérico desta Análise de Cluster é:")
print() # Este print sem argumento gera uma linha em branco
print(rotulos)
print() # Este print sem argumento gera uma linha em branco

# Vamos ver o exemplo através de um gráfico para melhor entendimento

#Importando biblioteca necessária para a plotagem do gráfico
import matplotlib.pyplot as plt

print("O resultado gráfico desta Análise de Cluster é:")
print() # Este print sem argumento gera uma linha em branco

# Criar uma figura e um eixo
fig, ax = plt.subplots()

# Adicionar os pontos ao eixo
ax.scatter(X[:,0], X[:,1], c=rotulos)

# Exibir o gráfico
plt.title('Análise de Cluster')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.show()