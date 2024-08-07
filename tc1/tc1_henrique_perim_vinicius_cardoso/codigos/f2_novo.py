# -*- coding: utf-8 -*-
"""f2_novo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lY-ZqQP01HTIAPdlI8v0rQvRQvFb_vPe

# Função $f_2$:

Minimizar a soma das distâncias euclidianas entre os clientes e seus respectivos pontos de acesso.

 Restrições:
- Raio de cobertura de cada PA: 70 metros
- Demanda máxima de cada PA: 54 Mbps
- Cobertura mínima de clientes: 95%
- Número máximo de PAs: 25

# Importanto bibliotecas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

"""# Plot dos clientes:"""

df_clients = pd.read_csv("clientes.csv")

def add_header(df_clients):
    header = ['x', 'y', "consumo"]

    df_clients.to_csv("clientes_v2.csv", header=header, index=False)

    return df_clients

df_clients = add_header(df_clients)

def plot_clients(df_clients):
    df_clients = pd.read_csv("clientes_v2.csv")

    df_clients.plot(x='x', y='y', kind="scatter", xlim=(-50, 450), ylim=(-50, 450))
    plt.show()

plot_clients(df_clients)

"""# Definição do problema"""

def problem_def():
    df_clients = pd.read_csv("clientes_v2.csv")

    return df_clients

df_clients = problem_def()
print(df_clients)

"""# Solução inicial

## Primeira opção:
Utilizando K-means

## Segunda opção:
Para cada cliente, compare a distância deste com os outros cliente em um raio de 70 metros e crie um grupo com 18, depois obtenha a coordenada mais próxima do centro desse grupo e adicione um PA. Quando todos os clientes forem atendidos, meça a área formada por cada 4 conjunto de PAs e adicione um novo PA no centro dessa área da maior para a menor, até completar 25 PAs.

"""

def initial_solution(df_clients):
    nr_clusters = 25

    # applying k-means algorithm
    kmeans = KMeans(n_clusters=nr_clusters)
    df_clients['cluster'] = kmeans.fit_predict(df_clients[['x', 'y']])

    df_pa = pd.DataFrame(kmeans.cluster_centers_, columns=['x','y'])
    df_pa['id_pa'] = df_pa.index
    df_pa['carga_disponivel'] =  [54]*len(df_pa)
    df_pa['x'] = df_pa['x'].apply(lambda x: round(x / 5) * 5)
    df_pa['y'] = df_pa['y'].apply(lambda y: round(y / 5) * 5)

    return df_clients, df_pa

df_clients, df_pa = initial_solution(df_clients)

"""## Plot da clusterização"""

def plot_clustering():
    # Plote os pontos coloridos pelos clusters
    plt.figure(figsize=(8, 6))
    colors = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', \
        'b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm','y', 'k'
    ]

    for cluster_num in range(nr_clusters):
        cluster_points = df_clients[df_clients['cluster'] == cluster_num]
        plt.scatter(cluster_points['x'], cluster_points['y'], s=30,
                    c=colors[cluster_num], label=f'Cluster {cluster_num}')

    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

plot_clustering()

"""## Plot da solução inicial"""

def plot_initial_solution():
    tamanho_espaco = 400
    raio_alcance = 70

    # localização de cada PA com seu raio de alcance
    plt.figure(figsize=(8, 8))
    plt.scatter(df_pa['x'], df_pa['y'], marker='o', color='blue', label='PAs')

    for i, row in df_pa.iterrows():
        circle = plt.Circle((row['x'], row['y']), raio_alcance, color='red', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.xlim(0, tamanho_espaco)
    plt.ylim(0, tamanho_espaco)
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Localização dos PAs (Solução Inicial)')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_initial_solution()

"""# Estruturas de vizinhança

1. Aproxima o PA com menor demanda em direção ao PA mais próximo com maior demanda
"""

# 1. Moves the AP with the lowest demand towards the nearest AP with the highest demand
def approach_higher_higher_demand():
    pass

"""2. Aproxima o PA com maior demanda em direção ao PA mais próximo com maior demanda"""

#2. Moves the AP with the highest demand towards the nearest AP with the highest demand
def approach_higher_lower_demand():
    pass

"""3. Desloca o PA que tenha parte da sua área de cobertura fora do plano para internamente ao plano"""

# 3. Move the AP that has part of its coverage area outside the plane to internal to the plane
def move_pas_further_in():
    pass

"""# BVNS"""

def shake(df_pa, k):
    if k == 1:
        df_pa = approach_higher_higher_demand(df_pa)
    elif k == 2:
        df_pa = approach_higher_lower_demand(df_pa)
    elif k == 3:
        df_pa = move_pas_further_in(df_pa)

    return df_pa

def neighborhoodChange(df_pa):
    # Apply K-Means again to redefine neighborhood

    df_clients = problem_def()
    df_clients, df_pa = initial_solution(df_clients)

    return df_pa
