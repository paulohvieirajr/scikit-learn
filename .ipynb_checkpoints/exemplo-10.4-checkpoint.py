#@title 10.3.4) Exemplo prático com Scikit_learn - Clusterização (2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importa a classe KMeans do módulo cluster da biblioteca scikit-learn.
# Essa classe é usada para realizar a clusterização com o algoritmo k-means.
from sklearn.cluster import KMeans
# Importa a classe StandardScaler do módulo preprocessing da biblioteca scikit-learn.
# Essa classe é usada para padronizar os dados antes da clusterização.
from sklearn.preprocessing import StandardScaler

# Cria um dicionário com dados fictícios de indicadores fundamentalistas para seis empresas diferentes.
dados = {
    'Empresa': ['Empresa A', 'Empresa B', 'Empresa C', 'Empresa D', 'Empresa E', 'Empresa F'],
    'Receita (milhões)': [100, 200, 150, 300, 250, 180],
    'Lucro Líquido (milhões)': [20, 30, 25, 40, 35, 28],
    'Margem de Lucro (%)': [20, 15, 16.67, 13.33, 14, 15.56],
}

# Converter os dados em um DataFrame
df = pd.DataFrame(dados)

# Seleciona apenas os indicadores fundamentalistas (Receita, Lucro Líquido e Margem de Lucro) para a clusterização.
# A coluna "Empresa" é removida porque não é relevante para a análise de padrões.
X = df.drop('Empresa', axis=1)

# Padronizar os dados para que tenham média zero e desvio padrão igual a um
# Cria uma instância do StandardScaler para padronizar os dados.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Realizar a clusterização com k-means (vamos assumir 2 clusters)
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar os rótulos dos clusters aos dados originais
df['Cluster'] = clusters

# Visualizar os resultados
print(df)

# Plotar os clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
plt.xlabel('Receita (padronizado)')
plt.ylabel('Lucro Líquido (padronizado)')
plt.title('Clusters de Empresas')
plt.show()

